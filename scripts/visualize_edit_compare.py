"""GEdit-Bench image-editing: DTR vs Baseline side-by-side generation.

Stage-1 of the rebuttal edit-comparison pipeline:

  1. Load GEdit-Bench (default ``stepfun-ai/GEdit-Bench``), filter by language
     (default EN).
  2. Sequentially load Baseline ckpt and DTR ckpt, generate one edited image
     per case with the SAME seed, save under
     ``{out}/generated/{base|dtr}/{task_type}/{language}/{key}.png``.
  3. Optionally (``--capture_attn``) hook the Sana DiT cross-attention via
     ``DiTCrossAttnContext`` from ``visualize_daam.py`` and dump per-keyword
     grids to ``{out}/attn_grids/{key}/daam_grids_{base,dtr}.npz``.
     The keyword for each case is read from a JSON produced by the scoring
     script (``--keywords_json`` -> ``{key: keyword}``).
  4. Optionally restrict the run to a user-supplied key list via
     ``--only_keys keys.json`` (JSON list of strings) — used in stage-3 to
     only re-generate the cases the user picked from the score CSV.

Designed so the SAME script handles BOTH:

  * full-set generation (no capture, all cases)             — stage 1
  * top-K re-generation with cross-attention capture        — stage 3
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor

# Make sibling script importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from unilip.constants import *  # noqa: F401,F403,E402
from unilip.model.builder import load_pretrained_model_general  # noqa: E402
from unilip.utils import disable_torch_init  # noqa: E402
from unilip.mm_utils import get_model_name_from_path  # noqa: E402
from unilip.pipeline_edit import CustomEditPipeline  # noqa: E402

# Reuse cross-attention machinery from the T2I script.
from visualize_daam import (  # noqa: E402
    DiTCrossAttnContext,
    aggregate_cross_attention,
    find_keyword_token_positions,
    keyword_grid,
)


# ────────────────────────────────────────────────────────────
# Prompt / seed helpers
# ────────────────────────────────────────────────────────────
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_edit_prompts(instruction: str) -> List[str]:
    """Mirror eval/GEdit-Bench/gedit.py:add_template, returning ``[pos, neg]``.

    The trailing ``<image>`` placeholder is replaced inside CustomEditPipeline
    by ``<img><IMG_CONTEXT>*256</img>`` — we keep it untouched here.
    """
    tpl = (
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n<img>"
    )
    return [
        tpl.format(input=f"Edit the image: {instruction}\n<image>"),
        tpl.format(input="Edit the image.\n<image>"),
    ]


# ────────────────────────────────────────────────────────────
# Single-case generation (with optional cross-attn capture)
# ────────────────────────────────────────────────────────────
def _save_grids(
    grids: Dict[str, np.ndarray],
    n_steps: int,
    n_layers: int,
    save_path: str,
    attn_qs: Optional[np.ndarray] = None,
    pos_prompt: Optional[str] = None,
    token_ids: Optional[List[int]] = None,
) -> None:
    """Persist per-keyword grids AND (when available) the raw aggregated
    cross-attention ``attn_qs`` of shape ``[Q, S]`` plus the prompt's token
    ids. The latter two let downstream tools (e.g. plot_edit_compare.py)
    re-derive arbitrary keyword grids without re-running the model.
    """
    payload: Dict[str, np.ndarray] = {
        "_meta_n_steps": np.array([n_steps], dtype=np.int32),
        "_meta_n_layers": np.array([n_layers], dtype=np.int32),
    }
    for i, (kw, g) in enumerate(grids.items()):
        payload[f"kw_{i}_{kw}"] = g.astype(np.float32)
    if attn_qs is not None:
        payload["attn_qs"] = attn_qs.astype(np.float32)
    if token_ids is not None:
        payload["token_ids"] = np.asarray(token_ids, dtype=np.int64)
    np.savez(save_path, **payload)
    if pos_prompt is not None:
        # Save the prompt as a sibling .json so it's human-readable and never
        # mangled by numpy's variable-length string handling.
        with open(os.path.splitext(save_path)[0] + "_prompt.json",
                  "w", encoding="utf-8") as f:
            json.dump({"pos_prompt": pos_prompt}, f, ensure_ascii=False, indent=2)


def _compute_keyword_grids(
    ctx: DiTCrossAttnContext,
    tokenizer,
    pos_prompt: str,
    keywords: List[str],
    step_window: Tuple[float, float],
    layer_indices: Optional[List[int]],
) -> Tuple[Dict[str, np.ndarray], int, int, Optional[np.ndarray], Optional[List[int]]]:
    """Aggregate captured cross-attention into per-keyword grids.

    Returns ``(grids, n_steps, n_layers, attn_qs_np, token_ids)`` where
    ``attn_qs_np`` is the [Q, S] aggregated attention as numpy float32 and
    ``token_ids`` is the prompt's token id list (for downstream re-matching).
    Both are ``None`` only when no attention was captured.
    """
    n_steps = len(ctx.capture.attn)
    n_layers = len(ctx.capture.attn[0]) if n_steps else 0
    grids: Dict[str, np.ndarray] = {}
    if n_steps == 0:
        return grids, n_steps, n_layers, None, None

    s0 = max(0, int(round(step_window[0] * n_steps)))
    s1 = max(s0 + 1, int(round(step_window[1] * n_steps)))
    step_indices = list(range(s0, s1))

    attn_qs = aggregate_cross_attention(ctx.capture, step_indices, layer_indices)
    Q, S = attn_qs.shape
    grid_side = int(round(math.sqrt(Q)))
    if grid_side * grid_side != Q:
        attn_qs = attn_qs[: grid_side * grid_side]
        grid_side = int(round(math.sqrt(attn_qs.shape[0])))

    attn_qs_np = attn_qs.detach().cpu().numpy() if hasattr(attn_qs, "detach") \
        else np.asarray(attn_qs)
    token_ids: Optional[List[int]] = None
    try:
        token_ids = tokenizer(pos_prompt, return_tensors="pt",
                              padding=False).input_ids[0].tolist()
    except Exception:
        token_ids = None

    for kw in keywords:
        positions = find_keyword_token_positions(tokenizer, pos_prompt, kw)
        positions = [p for p in positions if 0 <= p < S]
        if not positions:
            grids[kw] = np.zeros((grid_side, grid_side), dtype=np.float32)
            print(f"      keyword '{kw}': NOT FOUND in prompt tokens")
            continue
        grids[kw] = keyword_grid(attn_qs, positions, grid_size=grid_side)
        print(
            f"      keyword '{kw}': positions={positions} "
            f"max={grids[kw].max():.4f} mean={grids[kw].mean():.4f}"
        )
    return grids, n_steps, n_layers, attn_qs_np, token_ids


def _generate_single(
    model,
    tokenizer,
    pipe: CustomEditPipeline,
    instruction: str,
    input_image: Image.Image,
    seed: int,
    guidance_scale: float,
    capture_attn: bool,
    keywords: List[str],
    step_window: Tuple[float, float],
    layer_indices: Optional[List[int]],
) -> Tuple[Image.Image, Optional[Dict[str, np.ndarray]], int, int,
           Optional[np.ndarray], Optional[List[int]], Optional[str]]:
    """Run the edit pipeline once, optionally capturing DAAM grids.

    Returns ``(image, grids, n_steps, n_layers, attn_qs_np, token_ids, pos_prompt)``.
    The trailing three are ``None`` when ``capture_attn`` is False.
    """
    pos_prompt, neg_prompt = build_edit_prompts(instruction)
    set_global_seed(seed)
    gen = torch.Generator(device=model.device).manual_seed(seed)

    if capture_attn:
        with DiTCrossAttnContext(model) as ctx:
            edited = pipe(
                [pos_prompt, neg_prompt, input_image.convert("RGB")],
                guidance_scale=guidance_scale,
                generator=gen,
            )
        grids, n_steps, n_layers, attn_qs_np, token_ids = _compute_keyword_grids(
            ctx=ctx,
            tokenizer=tokenizer,
            pos_prompt=pos_prompt,
            keywords=keywords,
            step_window=step_window,
            layer_indices=layer_indices,
        )
        return edited, grids, n_steps, n_layers, attn_qs_np, token_ids, pos_prompt
    else:
        edited = pipe(
            [pos_prompt, neg_prompt, input_image.convert("RGB")],
            guidance_scale=guidance_scale,
            generator=gen,
        )
        return edited, None, 0, 0, None, None, None


# ────────────────────────────────────────────────────────────
# Per-checkpoint pass over the dataset
# ────────────────────────────────────────────────────────────
def _free_model(model) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_pass(
    model_path: str,
    tag: str,
    items: List[dict],
    out_root: str,
    seed: int,
    guidance_scale: float,
    capture_attn: bool,
    keywords_by_key: Dict[str, str],
    step_window: Tuple[float, float],
    layer_indices: Optional[List[int]],
    skip_existing: bool,
) -> None:
    """Generate edited images (and optional DAAM grids) for one checkpoint."""
    disable_torch_init()
    mp = os.path.expanduser(model_path)
    mname = get_model_name_from_path(mp)
    print(f"\n>>> [{tag.upper()}] loading model: {mp}")
    tokenizer, multi_model, _ = load_pretrained_model_general(
        "UniLIP_InternVLForCausalLM", mp, None, mname
    )
    multi_model.eval()
    image_processor = AutoProcessor.from_pretrained(
        multi_model.config.mllm_hf_path
    ).image_processor
    pipe = CustomEditPipeline(
        multimodal_encoder=multi_model,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    print(f">>> [{tag.upper()}] ready. {len(items)} items to generate. "
          f"capture_attn={capture_attn}")

    for idx, item in enumerate(tqdm(items, desc=f"edit[{tag}]")):
        key = item["key"]
        task_type = item["task_type"]
        language = item["instruction_language"]
        instruction = item["instruction"]

        out_dir = os.path.join(out_root, "generated", tag, task_type, language)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{key}.png")

        attn_dir = os.path.join(out_root, "attn_grids", key)
        attn_path = os.path.join(attn_dir, f"daam_grids_{tag}.npz")

        # If we don't need attention this round and the image already exists,
        # just skip. If we DO need attention, only skip when both files exist.
        img_exists = os.path.exists(out_path)
        attn_exists = os.path.exists(attn_path) if capture_attn else True
        if skip_existing and img_exists and attn_exists:
            continue

        kw_str = keywords_by_key.get(key, "")
        keywords = [k.strip() for k in kw_str.split(",") if k.strip()]
        if capture_attn and not keywords:
            print(f"   [{tag}|{key}] capture_attn requested but no keyword "
                  f"available — capturing anyway with empty keyword list.")

        try:
            (edited, grids, n_steps, n_layers,
             attn_qs_np, token_ids, pos_prompt) = _generate_single(
                model=multi_model,
                tokenizer=tokenizer,
                pipe=pipe,
                instruction=instruction,
                input_image=item["input_image_raw"],
                seed=seed,
                guidance_scale=guidance_scale,
                capture_attn=capture_attn,
                keywords=keywords,
                step_window=step_window,
                layer_indices=layer_indices,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"   [{tag}|{key}] ERROR: {exc!r}")
            import traceback
            traceback.print_exc()
            continue

        edited.save(out_path)

        if capture_attn and grids is not None:
            os.makedirs(attn_dir, exist_ok=True)
            _save_grids(
                grids=grids, n_steps=n_steps, n_layers=n_layers,
                save_path=attn_path,
                attn_qs=attn_qs_np,
                pos_prompt=pos_prompt,
                token_ids=token_ids,
            )

        if (idx + 1) % 25 == 0:
            print(f"   [{tag}] {idx + 1}/{len(items)} done.")

    _free_model(multi_model)
    del pipe, tokenizer, image_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f">>> [{tag.upper()}] pass finished.")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def _build_items(
    dataset_name: str,
    dataset_split: str,
    language: str,
    only_keys: Optional[List[str]],
    max_cases: int,
) -> List[dict]:
    ds = load_dataset(dataset_name)[dataset_split]
    items: List[dict] = []
    only_set = set(only_keys) if only_keys else None
    for item in ds:
        if language != "all" and item["instruction_language"] != language:
            continue
        if only_set is not None and item["key"] not in only_set:
            continue
        items.append(item)
        if max_cases > 0 and len(items) >= max_cases:
            break

    if only_set is not None:
        # Preserve the user-requested order if they passed a list.
        order = {k: i for i, k in enumerate(only_keys)}
        items.sort(key=lambda it: order.get(it["key"], 1_000_000))

    return items


_STOPWORDS = {
    # Function words.
    "the", "a", "an", "of", "in", "on", "with", "and", "or", "to", "for",
    "at", "by", "from", "as", "into", "onto", "over", "under", "this",
    "that", "these", "those", "it", "its", "be", "is", "are", "was", "were",
    "have", "has", "had", "do", "does", "did", "but", "if", "then", "than",
    "so", "such", "no", "not", "all", "any", "some", "more", "less",
    "please", "just", "very", "also", "only", "now", "still", "out", "up",
    "down", "off", "back", "again", "though", "while", "when", "where",
    # Prepositions / spatial relations.
    "behind", "front", "above", "below", "near", "next", "between", "around",
    "across", "along", "beside", "inside", "outside", "beneath", "underneath",
    "toward", "towards", "against", "among", "amongst", "before", "after",
    # Pronouns / determiners.
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "mine", "yours",
    # Generic 'image' meta-words: useless as visual keywords.
    "image", "picture", "photo", "scene", "version", "result",
    # Position words: too generic to be a useful keyword.
    "left", "right", "top", "bottom", "middle", "center", "centre", "side",
    # Generic edit verbs (bare-form, no -ing/-ed suffix to catch).
    # These dominate every prompt and are useless as visual keywords.
    "add", "remove", "delete", "replace", "change", "swap", "edit", "modify",
    "alter", "transform", "convert", "make", "let", "turn", "keep", "use",
    "put", "place", "set", "give", "take", "show", "draw", "paint", "fix",
    "move", "rotate", "shift", "scale", "resize", "crop", "fill", "erase",
    "apply", "create", "generate", "render", "adjust", "tweak", "refine",
    "swap", "include", "introduce", "drop",
    # Common color/attribute words used as adjectives — usually too generic
    # to be a salient noun keyword on their own. Comment out if you DO want
    # color heatmaps. Leaving in to focus heat-maps on objects.
    "small", "large", "big", "tiny", "huge",
}

# Lightweight verb suffix heuristics; cheap stand-in for a real POS tagger.
_VERB_SUFFIXES = ("ing", "ed", "ize", "ise", "ify")
# Likely-noun suffixes (after stopword filtering).
_NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "er", "or", "ist",
                  "ism", "ship", "hood", "ance", "ence")


def _word_count(prompt: str) -> int:
    import re
    return len(re.findall(r"[A-Za-z]+", prompt or ""))


def _extract_keywords(
    prompt: str, max_kw: int = 4,
) -> List[str]:
    """Pull a short ordered NOUN-only keyword list from an edit instruction.

    Heuristic, no external NLP deps. Verbs are explicitly excluded.
      * tokenize alphabetic words,
      * drop stopwords / pronouns / generic 'image' words / pure positions,
      * drop anything that LOOKS like a verb (``-ing/-ed/-ize/-ise/-ify``)
        — even though English does form noun gerunds, for keyword-attention
        visualization we prefer concrete object nouns,
      * prefer words with noun-like suffixes / plural -s / capitalized,
      * fall back to remaining content words (still verb-filtered) so we
        almost always get something for short prompts,
      * de-duplicate preserving first-seen order, cap to ``max_kw``.

    Returns lowercase list. Empty only when the prompt has no usable noun.
    """
    import re
    s = (prompt or "").strip()
    if not s:
        return []
    words = re.findall(r"[A-Za-z]+", s)

    primary: List[str] = []
    fallback: List[str] = []
    for w in words:
        wl = w.lower()
        if len(wl) < 3:
            continue
        if wl in _STOPWORDS:
            continue
        # Verb-suffix filter: '-ed' / '-ing' are too aggressive on short
        # words (e.g. 'red', 'bed', 'fed', 'sing' is rare; require min len).
        is_verb_like = False
        for suf in _VERB_SUFFIXES:
            if wl.endswith(suf) and len(wl) >= len(suf) + 3:
                is_verb_like = True
                break
        if is_verb_like:
            continue
        is_noun_like = (
            wl.endswith(_NOUN_SUFFIXES)
            or wl.endswith("s")          # plural
            or w[0].isupper()            # proper noun
        )
        if is_noun_like:
            primary.append(wl)
        else:
            fallback.append(wl)

    seen: set = set()
    ordered: List[str] = []
    for src in (primary, fallback):
        for w in src:
            if w in seen:
                continue
            seen.add(w)
            ordered.append(w)
            if len(ordered) >= max_kw:
                return ordered
    return ordered


def _build_items_sft(
    sft_tar_glob: str,
    sample_n: int,
    sample_seed: int,
    only_keys: Optional[List[str]],
    max_prompt_words: int = 0,
) -> List[dict]:
    """Sample N edit examples from the local SFT webdataset (.tar shards).

    Each shard yields {input.jpg, output.jpg, txt}. We stream-sample with
    reservoir sampling so memory stays bounded regardless of dataset size.
    The returned items match the GEdit schema consumed by ``_run_pass``:
    ``{key, task_type, instruction_language, instruction, input_image_raw}``.

    ``only_keys`` is honored just like in the GEdit branch — when provided
    we IGNORE the random sampling and instead pull exactly those keys
    (assumed to be ``sft_<6-digit-index>``).

    ``max_prompt_words`` (>0) filters samples whose instruction has more
    than this many alphabetic words BEFORE entering the reservoir, so the
    sampled distribution is uniform over short-prompt samples (not biased
    by the long-prompt majority). The key index ``cur_idx`` is only
    incremented for samples that pass the filter, keeping ``sft_XXXXXX``
    keys deterministic w.r.t. (glob, max_prompt_words).
    """
    import glob
    import io
    import tarfile

    tar_files = sorted(glob.glob(sft_tar_glob))
    if not tar_files:
        raise FileNotFoundError(
            f"No .tar shards matched glob: {sft_tar_glob}"
        )
    print(f">>> SFT: {len(tar_files)} tar shards found under {sft_tar_glob}")

    # Resolve only_keys -> wanted indices set if provided.
    wanted_idx: Optional[set] = None
    if only_keys:
        wanted_idx = set()
        for k in only_keys:
            if not k.startswith("sft_"):
                continue
            try:
                wanted_idx.add(int(k[len("sft_"):]))
            except ValueError:
                pass
        print(f">>> SFT: only_keys mode, {len(wanted_idx)} wanted indices.")

    rng = random.Random(sample_seed)
    reservoir: List[Tuple[int, dict]] = []  # (global_idx, item)
    targeted: Dict[int, dict] = {}          # idx -> item, for only_keys mode
    global_idx = 0

    for tar_path in tar_files:
        try:
            tf = tarfile.open(tar_path, "r")
        except Exception as e:
            print(f"   WARN: cannot open {tar_path}: {e}")
            continue
        # Group members by sample stem (everything before the first '.').
        members_by_stem: Dict[str, Dict[str, tarfile.TarInfo]] = {}
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            stem, _, ext = name.partition(".")
            if not ext:
                continue
            members_by_stem.setdefault(stem, {})[ext.lower()] = m

        for stem in sorted(members_by_stem.keys()):
            grp = members_by_stem[stem]
            # webdataset edit schema requires all three.
            need = ("input.jpg", "output.jpg", "txt")
            if not all(k in grp for k in need):
                continue

            # Pre-filter on prompt length: read txt FIRST (cheap), only
            # count this sample toward cur_idx / sampling if it survives.
            try:
                txt_bytes = tf.extractfile(grp["txt"]).read()
                instruction = txt_bytes.decode("utf-8", errors="replace").strip()
            except Exception as e:
                print(f"   WARN: bad txt in {tar_path}:{stem}: {e}")
                continue
            if max_prompt_words > 0 and _word_count(instruction) > max_prompt_words:
                continue

            cur_idx = global_idx
            global_idx += 1

            # Decide whether to materialize this sample.
            # Reservoir sampling (Algorithm R) over the FILTERED stream.
            keep_slot: Optional[int] = None
            if wanted_idx is not None:
                if cur_idx in wanted_idx:
                    keep_slot = -1  # sentinel: targeted mode
            else:
                if len(reservoir) < sample_n:
                    keep_slot = len(reservoir)
                else:
                    j = rng.randint(0, cur_idx)
                    if j < sample_n:
                        keep_slot = j

            if keep_slot is None:
                continue

            try:
                in_bytes = tf.extractfile(grp["input.jpg"]).read()
                in_pil = Image.open(io.BytesIO(in_bytes)).convert("RGB").copy()
            except Exception as e:
                print(f"   WARN: bad input.jpg in {tar_path}:{stem}: {e}")
                continue

            item = {
                "key": f"sft_{cur_idx:06d}",
                "task_type": "sft_edit",
                "instruction_language": "en",
                "instruction": instruction,
                "input_image_raw": in_pil,
                "_sft_shard": os.path.basename(tar_path),
                "_sft_stem": stem,
            }
            if wanted_idx is not None:
                targeted[cur_idx] = item
                if len(targeted) >= len(wanted_idx):
                    tf.close()
                    break  # done with this shard
            else:
                if keep_slot == len(reservoir):
                    reservoir.append((cur_idx, item))
                else:
                    reservoir[keep_slot] = (cur_idx, item)
        tf.close()
        if wanted_idx is not None and len(targeted) >= len(wanted_idx):
            break

    if wanted_idx is not None:
        items = [targeted[i] for i in sorted(targeted.keys())]
    else:
        reservoir.sort(key=lambda t: t[0])
        items = [it for _, it in reservoir]

    print(f">>> SFT: scanned {global_idx} samples, returning {len(items)} items.")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GEdit-Bench DTR vs Baseline edit generation"
                    " (with optional DAAM cross-attn capture)",
    )
    parser.add_argument("--baseline_model_path", required=True)
    parser.add_argument("--dtr_model_path", required=True)
    parser.add_argument("--out_root", default="results/vis_edit_compare")
    parser.add_argument(
        "--source", default="gedit", choices=["gedit", "sft"],
        help="Where edit cases come from: HF GEdit-Bench, or local SFT "
             "webdataset .tar shards.",
    )
    parser.add_argument("--dataset_name", default="stepfun-ai/GEdit-Bench")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument(
        "--language", default="en", choices=["all", "en", "cn"],
        help="Restrict to one language (GEdit only). Default 'en'.",
    )
    parser.add_argument(
        "--sft_tar_glob",
        default="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/"
                "my_unilip/data/edit_sft/*.tar",
        help="Glob for SFT edit webdataset .tar shards.",
    )
    parser.add_argument(
        "--sft_sample_n", type=int, default=50,
        help="Number of SFT samples to draw via reservoir sampling.",
    )
    parser.add_argument(
        "--sft_sample_seed", type=int, default=0,
        help="Reservoir-sampling RNG seed (use the same to reproduce).",
    )
    parser.add_argument(
        "--sft_max_prompt_words", type=int, default=10,
        help="Skip SFT samples whose instruction has more than this many "
             "alphabetic words BEFORE sampling. 0 = no limit.",
    )
    parser.add_argument(
        "--sft_max_keywords", type=int, default=4,
        help="When --capture_attn, max distinct keywords (nouns) to extract "
             "per SFT prompt for heat-map visualization.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument(
        "--max_cases", type=int, default=0,
        help="If >0, cap the number of cases (debugging).",
    )
    parser.add_argument(
        "--only_keys", default="",
        help="Path to JSON list of GEdit keys to restrict to. Used in stage 3.",
    )
    parser.add_argument(
        "--side", choices=["both", "base", "dtr"], default="both",
        help="Run only one side (useful when re-doing a single ckpt).",
    )

    # Cross-attn capture options
    parser.add_argument(
        "--capture_attn", action="store_true",
        help="Hook DiT cross-attn and dump per-keyword grids.",
    )
    parser.add_argument(
        "--keywords_json", default="",
        help="JSON dict {key: 'kw1,kw2'} produced by score_edit_compare.py."
             " Required when --capture_attn is set, otherwise ignored.",
    )
    parser.add_argument(
        "--step_window", default="0.4,0.6",
        help="Fractional denoising-step window to average over.",
    )
    parser.add_argument(
        "--layers", default="",
        help="Comma-separated DiT layer indices. Empty = all layers.",
    )

    parser.add_argument(
        "--no_skip_existing", action="store_true",
        help="Re-generate even if outputs already exist.",
    )
    args = parser.parse_args()

    s_lo, s_hi = (float(x) for x in args.step_window.split(","))
    step_window = (s_lo, s_hi)

    layer_indices: Optional[List[int]] = None
    if args.layers.strip():
        layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    only_keys: Optional[List[str]] = None
    if args.only_keys.strip():
        with open(args.only_keys, "r", encoding="utf-8") as f:
            only_keys = json.load(f)
        if not isinstance(only_keys, list):
            raise ValueError(f"--only_keys must point to a JSON list, got {type(only_keys)}")
        print(f">>> only_keys: {len(only_keys)} keys loaded from {args.only_keys}")

    keywords_by_key: Dict[str, str] = {}
    if args.capture_attn:
        if not args.keywords_json:
            print(">>> WARNING: --capture_attn set but no --keywords_json; "
                  "will capture attn but produce zero-grid for every keyword.")
        else:
            with open(args.keywords_json, "r", encoding="utf-8") as f:
                keywords_by_key = json.load(f)
            print(f">>> keywords loaded for {len(keywords_by_key)} keys.")

    if args.source == "gedit":
        items = _build_items(
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            language=args.language,
            only_keys=only_keys,
            max_cases=args.max_cases,
        )
        print(f">>> {len(items)} GEdit-Bench cases selected "
              f"(language={args.language}, only_keys={'yes' if only_keys else 'no'}).")
    else:
        items = _build_items_sft(
            sft_tar_glob=args.sft_tar_glob,
            sample_n=args.sft_sample_n,
            sample_seed=args.sft_sample_seed,
            only_keys=only_keys,
            max_prompt_words=args.sft_max_prompt_words,
        )
        if args.max_cases > 0:
            items = items[: args.max_cases]
        print(f">>> {len(items)} SFT edit cases selected "
              f"(seed={args.sft_sample_seed}, "
              f"max_prompt_words={args.sft_max_prompt_words}, "
              f"only_keys={'yes' if only_keys else 'no'}).")
        # SFT mode: auto-build NOUN keywords (heuristic) so capture works
        # without an external keywords_json. User-supplied --keywords_json
        # still wins (we only fill keys that aren't already present).
        # Multiple keywords per case are joined with ',' so the existing
        # keywords_by_key parsing in _run_pass picks them all up.
        if args.capture_attn:
            n_filled = 0
            for it in items:
                k = it["key"]
                if k in keywords_by_key and keywords_by_key[k].strip():
                    continue
                kws = _extract_keywords(it["instruction"],
                                        max_kw=args.sft_max_keywords)
                if kws:
                    keywords_by_key[k] = ",".join(kws)
                    n_filled += 1
            print(f">>> SFT: auto-filled NOUN keywords for {n_filled} items "
                  f"(now {len(keywords_by_key)} keys have a keyword list).")
            # Persist the auto-keywords for reproducibility / re-use.
            os.makedirs(args.out_root, exist_ok=True)
            kw_dump = os.path.join(args.out_root, "keywords_auto.json")
            with open(kw_dump, "w", encoding="utf-8") as f:
                json.dump(keywords_by_key, f, ensure_ascii=False, indent=2)
            print(f">>> SFT: auto-keywords saved to {kw_dump}")

    # Save the resolved key list as a reproducibility receipt.
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "selected_keys.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"key": it["key"], "task_type": it["task_type"],
              "instruction_language": it["instruction_language"],
              "instruction": it["instruction"],
              "_sft_shard": it.get("_sft_shard"),
              "_sft_stem": it.get("_sft_stem"),
              "keyword": keywords_by_key.get(it["key"], "")}
             for it in items],
            f, ensure_ascii=False, indent=2,
        )

    skip_existing = not args.no_skip_existing

    if args.side in ("both", "base"):
        _run_pass(
            model_path=args.baseline_model_path,
            tag="base",
            items=items,
            out_root=args.out_root,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            capture_attn=args.capture_attn,
            keywords_by_key=keywords_by_key,
            step_window=step_window,
            layer_indices=layer_indices,
            skip_existing=skip_existing,
        )

    if args.side in ("both", "dtr"):
        _run_pass(
            model_path=args.dtr_model_path,
            tag="dtr",
            items=items,
            out_root=args.out_root,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            capture_attn=args.capture_attn,
            keywords_by_key=keywords_by_key,
            step_window=step_window,
            layer_indices=layer_indices,
            skip_existing=skip_existing,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()

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
) -> None:
    payload: Dict[str, np.ndarray] = {
        "_meta_n_steps": np.array([n_steps], dtype=np.int32),
        "_meta_n_layers": np.array([n_layers], dtype=np.int32),
    }
    for i, (kw, g) in enumerate(grids.items()):
        payload[f"kw_{i}_{kw}"] = g.astype(np.float32)
    np.savez(save_path, **payload)


def _compute_keyword_grids(
    ctx: DiTCrossAttnContext,
    tokenizer,
    pos_prompt: str,
    keywords: List[str],
    step_window: Tuple[float, float],
    layer_indices: Optional[List[int]],
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Aggregate captured cross-attention into per-keyword grids."""
    n_steps = len(ctx.capture.attn)
    n_layers = len(ctx.capture.attn[0]) if n_steps else 0
    grids: Dict[str, np.ndarray] = {}
    if n_steps == 0 or not keywords:
        return grids, n_steps, n_layers

    s0 = max(0, int(round(step_window[0] * n_steps)))
    s1 = max(s0 + 1, int(round(step_window[1] * n_steps)))
    step_indices = list(range(s0, s1))

    attn_qs = aggregate_cross_attention(ctx.capture, step_indices, layer_indices)
    Q, S = attn_qs.shape
    grid_side = int(round(math.sqrt(Q)))
    if grid_side * grid_side != Q:
        attn_qs = attn_qs[: grid_side * grid_side]
        grid_side = int(round(math.sqrt(attn_qs.shape[0])))

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
    return grids, n_steps, n_layers


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
) -> Tuple[Image.Image, Optional[Dict[str, np.ndarray]], int, int]:
    """Run the edit pipeline once, optionally capturing DAAM grids."""
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
        grids, n_steps, n_layers = _compute_keyword_grids(
            ctx=ctx,
            tokenizer=tokenizer,
            pos_prompt=pos_prompt,
            keywords=keywords,
            step_window=step_window,
            layer_indices=layer_indices,
        )
        return edited, grids, n_steps, n_layers
    else:
        edited = pipe(
            [pos_prompt, neg_prompt, input_image.convert("RGB")],
            guidance_scale=guidance_scale,
            generator=gen,
        )
        return edited, None, 0, 0


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
            edited, grids, n_steps, n_layers = _generate_single(
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
            _save_grids(grids, n_steps, n_layers, attn_path)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GEdit-Bench DTR vs Baseline edit generation"
                    " (with optional DAAM cross-attn capture)",
    )
    parser.add_argument("--baseline_model_path", required=True)
    parser.add_argument("--dtr_model_path", required=True)
    parser.add_argument("--out_root", default="results/vis_edit_compare")
    parser.add_argument("--dataset_name", default="stepfun-ai/GEdit-Bench")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument(
        "--language", default="en", choices=["all", "en", "cn"],
        help="Restrict to one language. Default 'en' for the rebuttal.",
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

    items = _build_items(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        language=args.language,
        only_keys=only_keys,
        max_cases=args.max_cases,
    )
    print(f">>> {len(items)} GEdit-Bench cases selected "
          f"(language={args.language}, only_keys={'yes' if only_keys else 'no'}).")

    # Save the resolved key list as a reproducibility receipt.
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "selected_keys.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"key": it["key"], "task_type": it["task_type"],
              "instruction_language": it["instruction_language"],
              "instruction": it["instruction"]}
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

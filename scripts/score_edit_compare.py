"""GEdit-Bench DTR vs Baseline scoring with Qwen2.5-VL-72B-AWQ (VIEScore tie).

Stage-2 of the rebuttal edit-comparison pipeline. Reads the images written
by ``visualize_edit_compare.py`` and produces:

  * ``scores/base_scores.json`` — {key: {SC, PQ, O, raw_SC, raw_PQ, ...}}
  * ``scores/dtr_scores.json``  — same shape, for DTR
  * ``scores/edited_objects.json`` — {key: "kw1, kw2"} (1 Qwen call/case,
    asks Qwen to extract the noun phrase that the instruction edits, using
    only the input image + instruction; same for both ckpts)
  * ``scores/score_diff.csv`` — flat CSV sorted by delta_O desc, contains
    every metadata field the user asked for (qwen_model, qwen_seed,
    image paths, …) so the rebuttal numbers are fully reproducible.
  * ``scores/selected_topk.json`` — top-K keys by delta_O (default K=8),
    consumed by stage-3 (re-generate with --capture_attn).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ────────────────────────────────────────────────────────────
# viescore lives in Step1X-Edit/GEdit-Bench with relative imports
# (sys.path.insert(0, 'viescore') at the top of __init__.py),
# so we MUST chdir there before importing it.
# ────────────────────────────────────────────────────────────
def _import_viescore(repo_root: str):
    vie_dir = os.path.join(repo_root, "Step1X-Edit", "GEdit-Bench")
    if not os.path.isdir(vie_dir):
        raise FileNotFoundError(f"viescore dir not found: {vie_dir}")
    os.chdir(vie_dir)  # required by the upstream sys.path hack
    sys.path.insert(0, vie_dir)
    from viescore import VIEScore  # noqa: E402
    from viescore.mllm_tools.qwen25vl_eval import Qwen25VL  # noqa: E402
    return VIEScore, Qwen25VL


# ────────────────────────────────────────────────────────────
# Qwen helpers
# ────────────────────────────────────────────────────────────
_OBJECT_PROMPT_TEMPLATE = (
    "You are given an image and an editing instruction. Identify the main "
    "object(s) in the image that the instruction asks to modify, add, "
    "remove, or replace.\n\n"
    "Instruction: {instruction}\n\n"
    "Reply with ONE short noun phrase in English (1-3 words, no article, "
    "no quotes, no extra text). If multiple distinct objects are involved, "
    "join them with a comma (max 2). Examples: 'cat', 'red car', "
    "'cat, dog'."
)


def _extract_edited_object(
    qwen,
    input_image: Image.Image,
    instruction: str,
) -> str:
    """Single Qwen call: ask for the noun phrase that names the edit target."""
    prompt = _OBJECT_PROMPT_TEMPLATE.format(instruction=instruction.strip())
    msgs = qwen.prepare_prompt([input_image.convert("RGB")], prompt)
    raw = qwen.get_parsed_output(msgs).strip()

    # Sanitise: keep first line, strip quotes / punctuation tails.
    line = raw.splitlines()[0].strip() if raw else ""
    line = line.strip("\"'`. \t")
    # Normalise whitespace.
    line = re.sub(r"\s+", " ", line)
    # Reject obviously bad outputs.
    if not line or len(line) > 64:
        return ""
    return line


def _calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    import math
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return int(width), int(height)


def _resize_for_score(img: Image.Image) -> Image.Image:
    w, h = _calculate_dimensions(512 * 512, img.width / max(img.height, 1))
    return img.resize((max(w, 1), max(h, 1)))


def _safe_evaluate(
    vie_score,
    pil_input: Image.Image,
    pil_edited: Image.Image,
    instruction: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Evaluate one (input, edited, instruction) triple. Returns dict with
    SC/PQ/O scalar scores and raw response strings, or None on failure."""
    import time
    for attempt in range(max_retries):
        try:
            inp_resized = _resize_for_score(pil_input.convert("RGB"))
            edt_resized = _resize_for_score(pil_edited.convert("RGB"))

            # Mirror VIEScore.evaluate but also keep raw responses.
            if vie_score.task == "tie":
                _SC_prompt = vie_score.SC_prompt.replace("<instruction>", instruction)
            else:
                _SC_prompt = vie_score.SC_prompt.replace("<prompt>", instruction)

            image_prompts = [inp_resized, edt_resized]
            SC_msgs = vie_score.model.prepare_prompt(image_prompts, _SC_prompt)
            PQ_msgs = vie_score.model.prepare_prompt(image_prompts[-1], vie_score.PQ_prompt)

            from utils import mllm_output_to_dict  # provided by viescore at chdir-time
            import math as _math

            raw_SC = vie_score.model.get_parsed_output(SC_msgs)
            raw_PQ = vie_score.model.get_parsed_output(PQ_msgs)

            SC_dict = mllm_output_to_dict(raw_SC, give_up_parsing=False)
            PQ_dict = mllm_output_to_dict(raw_PQ, give_up_parsing=False)
            if SC_dict is False or PQ_dict is False:
                # Retry once with give_up=True so we still produce a number.
                SC_dict = mllm_output_to_dict(raw_SC, give_up_parsing=True) or SC_dict
                PQ_dict = mllm_output_to_dict(raw_PQ, give_up_parsing=True) or PQ_dict
            if SC_dict in ("rate_limit_exceeded",) or PQ_dict in ("rate_limit_exceeded",):
                raise RuntimeError("rate_limit_exceeded")
            if SC_dict is False or PQ_dict is False:
                raise RuntimeError(f"parse failed; SC={raw_SC!r} PQ={raw_PQ!r}")

            SC_score = float(min(SC_dict["score"]))
            PQ_score = float(min(PQ_dict["score"]))
            O_score = float(_math.sqrt(max(SC_score, 0.0) * max(PQ_score, 0.0)))
            return {
                "SC": SC_score,
                "PQ": PQ_score,
                "O": O_score,
                "SC_all": SC_dict["score"],
                "PQ_all": PQ_dict["score"],
                "raw_SC": raw_SC,
                "raw_PQ": raw_PQ,
            }
        except Exception as exc:  # noqa: BLE001
            wait = 2 * (attempt + 1)
            print(f"      evaluate attempt {attempt+1}/{max_retries} failed: {exc!r} "
                  f"(sleep {wait}s)")
            time.sleep(wait)
    return None


# ────────────────────────────────────────────────────────────
# Path helpers
# ────────────────────────────────────────────────────────────
def _img_path(out_root: str, tag: str, task: str, lang: str, key: str) -> str:
    return os.path.join(out_root, "generated", tag, task, lang, f"{key}.png")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Baseline vs DTR edited images with Qwen2.5-VL-72B-AWQ "
                    "(VIEScore tie) and dump a flat CSV ranked by delta_O.",
    )
    parser.add_argument("--out_root", required=True,
                        help="Same --out_root used by visualize_edit_compare.py.")
    parser.add_argument("--repo_root", default="",
                        help="Path to my_unilip root (defaults to parent of scripts/).")
    parser.add_argument("--dataset_name", default="stepfun-ai/GEdit-Bench")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--language", default="en", choices=["all", "en", "cn"])
    parser.add_argument("--qwen_seed", type=int, default=42,
                        help="Seed used inside Qwen25VL.get_parsed_output.")
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
                        help="Pinned for the CSV (the actual model id is hard-coded "
                             "in viescore.mllm_tools.qwen25vl_eval).")
    parser.add_argument("--top_k", type=int, default=8,
                        help="Number of cases to dump into selected_topk.json.")
    parser.add_argument("--max_cases", type=int, default=0,
                        help="Cap evaluations (debug). 0 = no cap.")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="Re-score keys already present in scores JSONs.")
    args = parser.parse_args()

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    out_root = os.path.abspath(args.out_root)
    scores_dir = os.path.join(out_root, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    base_scores_path = os.path.join(scores_dir, "base_scores.json")
    dtr_scores_path = os.path.join(scores_dir, "dtr_scores.json")
    objects_path = os.path.join(scores_dir, "edited_objects.json")
    csv_path = os.path.join(scores_dir, "score_diff.csv")
    topk_path = os.path.join(scores_dir, "selected_topk.json")

    # 1) Load dataset (we need input_image_raw for both Qwen + the CSV).
    print(f">>> Loading dataset {args.dataset_name} (split={args.dataset_split}) …")
    ds = load_dataset(args.dataset_name)[args.dataset_split]
    items = []
    for item in ds:
        if args.language != "all" and item["instruction_language"] != args.language:
            continue
        items.append(item)
        if args.max_cases > 0 and len(items) >= args.max_cases:
            break
    by_key = {it["key"]: it for it in items}
    print(f">>> {len(items)} cases (language={args.language}).")

    # 2) Filter to cases for which BOTH base and DTR images exist.
    pending: List[dict] = []
    missing = 0
    for it in items:
        bp = _img_path(out_root, "base", it["task_type"], it["instruction_language"], it["key"])
        dp = _img_path(out_root, "dtr",  it["task_type"], it["instruction_language"], it["key"])
        if not (os.path.exists(bp) and os.path.exists(dp)):
            missing += 1
            continue
        pending.append(it)
    print(f">>> {len(pending)} cases have both base and dtr images "
          f"(skipped {missing} missing).")

    # 3) Load existing partial outputs for resume.
    base_scores: Dict[str, Any] = {}
    dtr_scores: Dict[str, Any] = {}
    edited_objects: Dict[str, str] = {}
    if os.path.exists(base_scores_path):
        with open(base_scores_path, "r", encoding="utf-8") as f:
            base_scores = json.load(f)
    if os.path.exists(dtr_scores_path):
        with open(dtr_scores_path, "r", encoding="utf-8") as f:
            dtr_scores = json.load(f)
    if os.path.exists(objects_path):
        with open(objects_path, "r", encoding="utf-8") as f:
            edited_objects = json.load(f)

    # 4) Import VIEScore (chdir into Step1X-Edit/GEdit-Bench under the hood).
    print(">>> Importing VIEScore + Qwen2.5-VL-72B-AWQ (this loads the model) …")
    VIEScore, _Qwen25VL = _import_viescore(repo_root)
    vie_score = VIEScore(backbone="qwen25vl", task="tie", key_path=None)
    qwen = vie_score.model  # reuse the loaded Qwen for object extraction
    print(">>> VIEScore ready.")

    # 5) Iterate cases.
    skip_existing = not args.no_skip_existing
    for idx, it in enumerate(tqdm(pending, desc="score")):
        key = it["key"]
        task = it["task_type"]
        lang = it["instruction_language"]
        instruction = it["instruction"]
        input_img = it["input_image_raw"]

        bp = _img_path(out_root, "base", task, lang, key)
        dp = _img_path(out_root, "dtr",  task, lang, key)

        # 5a) edited_object — once per case (instruction-only signal).
        if not skip_existing or key not in edited_objects:
            try:
                eo = _extract_edited_object(qwen, input_img, instruction)
            except Exception as exc:  # noqa: BLE001
                print(f"   [{key}] extract_edited_object failed: {exc!r}")
                eo = ""
            edited_objects[key] = eo
            if (idx + 1) % 10 == 0:
                with open(objects_path, "w", encoding="utf-8") as f:
                    json.dump(edited_objects, f, ensure_ascii=False, indent=2)

        # 5b) baseline score
        if not skip_existing or key not in base_scores:
            try:
                base_pil = Image.open(bp)
            except Exception as exc:  # noqa: BLE001
                print(f"   [{key}] cannot open base image {bp}: {exc!r}")
                continue
            r = _safe_evaluate(vie_score, input_img, base_pil, instruction)
            if r is not None:
                r.update({
                    "task_type": task,
                    "instruction_language": lang,
                    "instruction": instruction,
                    "image_path": bp,
                })
                base_scores[key] = r

        # 5c) dtr score
        if not skip_existing or key not in dtr_scores:
            try:
                dtr_pil = Image.open(dp)
            except Exception as exc:  # noqa: BLE001
                print(f"   [{key}] cannot open dtr image {dp}: {exc!r}")
                continue
            r = _safe_evaluate(vie_score, input_img, dtr_pil, instruction)
            if r is not None:
                r.update({
                    "task_type": task,
                    "instruction_language": lang,
                    "instruction": instruction,
                    "image_path": dp,
                })
                dtr_scores[key] = r

        # Periodic checkpoint to disk (resume safety).
        if (idx + 1) % 10 == 0:
            with open(base_scores_path, "w", encoding="utf-8") as f:
                json.dump(base_scores, f, ensure_ascii=False, indent=2)
            with open(dtr_scores_path, "w", encoding="utf-8") as f:
                json.dump(dtr_scores, f, ensure_ascii=False, indent=2)

    # 6) Final dump.
    with open(base_scores_path, "w", encoding="utf-8") as f:
        json.dump(base_scores, f, ensure_ascii=False, indent=2)
    with open(dtr_scores_path, "w", encoding="utf-8") as f:
        json.dump(dtr_scores, f, ensure_ascii=False, indent=2)
    with open(objects_path, "w", encoding="utf-8") as f:
        json.dump(edited_objects, f, ensure_ascii=False, indent=2)

    # 7) Build the flat CSV.
    rows: List[Dict[str, Any]] = []
    for key in base_scores.keys() & dtr_scores.keys():
        b = base_scores[key]
        d = dtr_scores[key]
        it = by_key.get(key, {})
        bp = _img_path(out_root, "base", b["task_type"], b["instruction_language"], key)
        dp = _img_path(out_root, "dtr",  b["task_type"], b["instruction_language"], key)
        rows.append({
            "key": key,
            "task_type": b["task_type"],
            "instruction_language": b["instruction_language"],
            "instruction": b["instruction"],
            "edited_object": edited_objects.get(key, ""),
            "base_image_path": bp,
            "dtr_image_path": dp,
            "input_image_dataset_key": key,  # raw image lives only in HF dataset
            "base_SC": b["SC"], "dtr_SC": d["SC"],
            "base_PQ": b["PQ"], "dtr_PQ": d["PQ"],
            "base_O":  b["O"],  "dtr_O":  d["O"],
            "delta_SC": d["SC"] - b["SC"],
            "delta_PQ": d["PQ"] - b["PQ"],
            "delta_O":  d["O"]  - b["O"],
            "qwen_model": args.qwen_model,
            "qwen_seed": args.qwen_seed,
            "viescore_task": "tie",
        })
    rows.sort(key=lambda r: r["delta_O"], reverse=True)

    fieldnames = [
        "key", "task_type", "instruction_language", "instruction",
        "edited_object",
        "base_image_path", "dtr_image_path", "input_image_dataset_key",
        "base_SC", "dtr_SC", "base_PQ", "dtr_PQ", "base_O", "dtr_O",
        "delta_SC", "delta_PQ", "delta_O",
        "qwen_model", "qwen_seed", "viescore_task",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f">>> Wrote {csv_path} ({len(rows)} rows).")

    # 8) Top-K candidate keys (DTR wins by largest delta_O, AND has a non-empty
    #    edited_object so stage-3 can actually draw a heat-map).
    candidates = [r for r in rows if r["edited_object"] and r["delta_O"] > 0]
    topk = candidates[: args.top_k]
    with open(topk_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "keys": [r["key"] for r in topk],
                "rows": topk,
                "k": args.top_k,
                "criterion": "delta_O > 0 AND edited_object != ''",
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f">>> Wrote {topk_path} (top-{args.top_k}).")

    # Aggregate sanity stats.
    if rows:
        import statistics as _st
        def _mean(xs): return _st.mean(xs) if xs else float("nan")
        print(
            f"\nSummary over {len(rows)} cases:\n"
            f"  base_O  mean={_mean([r['base_O']  for r in rows]):.3f}\n"
            f"  dtr_O   mean={_mean([r['dtr_O']   for r in rows]):.3f}\n"
            f"  delta_O mean={_mean([r['delta_O'] for r in rows]):+.3f}\n"
            f"  DTR>Base count: {sum(1 for r in rows if r['delta_O']>0)}\n"
        )

    print("All done.")


if __name__ == "__main__":
    main()

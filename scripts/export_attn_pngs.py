"""Export per-key PNGs for paper figures.

For each requested key under ``--out_root`` (typically results/vis_sft_edit),
writes the following standalone PNGs (no figure axes / matplotlib chrome):

  ``{out_root}/png_export/{key}/input.png``           — input image, resize 448x448
  ``{out_root}/png_export/{key}/edited_base.png``     — copy of base edit (already 448)
  ``{out_root}/png_export/{key}/edited_dtr.png``      — copy of dtr  edit (already 448)
  ``{out_root}/png_export/{key}/attn_<kw>_base.png``  — base attn overlay on edited_base
  ``{out_root}/png_export/{key}/attn_<kw>_dtr.png``   — dtr  attn overlay on edited_dtr

When ``--out_root_dtr2`` is supplied (a sibling pipeline output dir, e.g.
``results/vis_sft_edit_3b`` produced by running visualize_edit_compare.py with
``base=dtr=UniLIP-3B``), the export also writes:

  ``{out_root}/png_export/{key}/edited_dtr2.png``     — dtr2 (e.g. 3B) edit
  ``{out_root}/png_export/{key}/attn_<kw>_dtr2.png``  — dtr2 attn overlay

Per-keyword vmin/vmax is then shared across all three checkpoints.

The input is **resized to 448x448 with PIL.Image.LANCZOS** (no aspect-ratio
preservation) to exactly match what the edit pipeline feeds the model
(``unilip/pipeline_edit.py`` line 66: ``input_image.resize((448, 448))``).
This way input and edited PNGs share identical dimensions and can be placed
side-by-side in the paper without cropping.

Each ``attn_*`` PNG is the corresponding edit image with a jet-colormap
heat-map blended on top using the same ``alpha`` and per-keyword shared
``vmin/vmax`` as ``plot_sft_edit.py``, so the colour intensity is directly
comparable across the two checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from visualize_daam import _upsample_overlay  # noqa: E402
from plot_edit_compare import _resolve_grid_for_keyword  # noqa: E402
from plot_sft_edit import _ShardCache  # noqa: E402


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _safe_name(s: str) -> str:
    """Filesystem-safe slug for keyword strings."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "kw"


def _center_crop_pct(pil: Image.Image, pct: float) -> Image.Image:
    """Crop ``pct`` percent off each side (top/bottom/left/right).

    pct=10 -> keep central 80% region; pct=20 -> keep central 60% region.
    """
    if pct <= 0:
        return pil
    W, H = pil.size
    dx = int(round(W * pct / 100.0))
    dy = int(round(H * pct / 100.0))
    return pil.crop((dx, dy, W - dx, H - dy))


def _save_with_crops(
    pil: Image.Image,
    save_dir: str,
    filename: str,
    crop_pcts: List[int],
) -> None:
    """Save ``pil`` to ``save_dir/filename`` plus one copy per crop_pct under
    ``save_dir/crop{pct}/{filename}``. pct==0 is implicit (the un-cropped one).
    """
    pil.save(os.path.join(save_dir, filename))
    for pct in crop_pcts:
        if pct == 0:
            continue
        sub = os.path.join(save_dir, f"crop{pct}")
        os.makedirs(sub, exist_ok=True)
        _center_crop_pct(pil, pct).save(os.path.join(sub, filename))


def _overlay_pil(
    base_img: np.ndarray,
    heat: Optional[np.ndarray],
    vmin: float,
    vmax: float,
    cmap_name: str = "jet",
    alpha: float = 0.55,
) -> Tuple[Image.Image, bool]:
    """Blend ``heat`` (already upsampled to base_img H,W) on ``base_img`` and
    return a PIL image of the same pixel dims as ``base_img``.

    Second tuple element is False if heat is None / all-zero (in which case
    the returned PIL is just the bare base image so downstream paper layout
    stays intact)."""
    base_uint8 = (np.clip(base_img, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    if heat is None or heat.sum() <= 1e-12:
        return Image.fromarray(base_uint8), False

    denom = max(vmax - vmin, 1e-8)
    heat_n = np.clip((heat - vmin) / denom, 0.0, 1.0)

    cmap = cm.get_cmap(cmap_name)
    heat_rgb = cmap(heat_n)[..., :3]  # (H, W, 3) float
    heat_uint8 = (heat_rgb * 255.0).round().astype(np.uint8)

    blended = (alpha * heat_uint8 + (1.0 - alpha) * base_uint8).round().astype(np.uint8)
    return Image.fromarray(blended), True


def _load_keywords(
    out_root: str,
    key: str,
    user_keywords: Optional[List[str]] = None,
) -> List[str]:
    """Resolve keyword list for a key.

    Priority:
      1. ``user_keywords`` if non-empty (per-key override from CLI).
      2. ``selected_keys.json`` row's ``"keyword"`` field (comma-sep).
      3. ``keywords_auto.json`` lookup (if present).
    """
    if user_keywords:
        return user_keywords

    sel_path = os.path.join(out_root, "selected_keys.json")
    if os.path.exists(sel_path):
        with open(sel_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            if r.get("key") == key:
                kws = r.get("keyword", "") or r.get("keywords", "")
                return [k.strip() for k in kws.split(",") if k.strip()]

    kw_auto = os.path.join(out_root, "keywords_auto.json")
    if os.path.exists(kw_auto):
        with open(kw_auto, "r", encoding="utf-8") as f:
            auto = json.load(f)
        v = auto.get(key)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
    return []


def _shard_lookup(out_root: str, key: str) -> Optional[Tuple[str, str]]:
    """Return (shard_name, stem) for ``key`` from selected_keys.json."""
    sel_path = os.path.join(out_root, "selected_keys.json")
    if not os.path.exists(sel_path):
        return None
    with open(sel_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        if r.get("key") == key:
            return r.get("_sft_shard"), r.get("_sft_stem")
    return None


def _row_lookup(out_root: str, key: str) -> Dict:
    sel_path = os.path.join(out_root, "selected_keys.json")
    if not os.path.exists(sel_path):
        return {}
    with open(sel_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        if r.get("key") == key:
            return r
    return {}


# ────────────────────────────────────────────────────────────
# Per-key export
# ────────────────────────────────────────────────────────────
def _read_dtr2_assets(
    out_root_dtr2: str,
    out_root: str,
    key: str,
    task_type: str,
    language: str,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Locate dtr2 (e.g. 3B) edited PNG and attn npz.

    Two layouts supported:

    * **Same-root** (recommended, used by the 3-way driver): dtr2 lives next
      to base/dtr inside the SAME out_root, under ``generated/dtr2/`` and
      ``attn_grids/{key}/daam_grids_dtr2.npz``. Auto-detected when
      ``out_root_dtr2`` is empty.
    * **Sibling-root** (legacy): ``out_root_dtr2`` is a separate pipeline dir
      that was launched with ``base=dtr=3B``, so we read ``generated/dtr/``
      and ``daam_grids_dtr.npz`` from it.
    """
    if out_root_dtr2:
        edit_path = os.path.join(out_root_dtr2, "generated", "dtr",
                                 task_type, language, f"{key}.png")
        npz_path = os.path.join(out_root_dtr2, "attn_grids", key,
                                "daam_grids_dtr.npz")
    else:
        edit_path = os.path.join(out_root, "generated", "dtr2",
                                 task_type, language, f"{key}.png")
        npz_path = os.path.join(out_root, "attn_grids", key,
                                "daam_grids_dtr2.npz")
    pil = Image.open(edit_path).convert("RGB") if os.path.exists(edit_path) else None
    return pil, (npz_path if os.path.exists(npz_path) else None)


def export_one_key(
    *,
    out_root: str,
    key: str,
    shard_cache: _ShardCache,
    user_keywords: Optional[List[str]],
    target_size: Tuple[int, int],
    cmap: str,
    alpha: float,
    tokenizer,
    out_root_dtr2: str = "",
    crop_pcts: Optional[List[int]] = None,
) -> bool:
    """Export PNGs for a single key. Returns True on full success."""
    save_dir = os.path.join(out_root, "png_export", key)
    os.makedirs(save_dir, exist_ok=True)
    crop_pcts = crop_pcts or []

    row = _row_lookup(out_root, key)
    task_type = row.get("task_type", "sft_edit")
    language = row.get("instruction_language", "en")

    # ── 1) Input image: pull from tar shard, resize to target (default 448).
    sh = _shard_lookup(out_root, key)
    if not sh or not sh[0] or not sh[1]:
        print(f"[{key}] FAIL: shard/stem missing in selected_keys.json")
        return False
    shard_name, stem = sh
    input_pil = shard_cache.load_input(shard_name, stem)
    if input_pil is None:
        print(f"[{key}] FAIL: input image not loadable from {shard_name}:{stem}")
        return False
    input_pil = input_pil.convert("RGB").resize(target_size, Image.LANCZOS)
    _save_with_crops(input_pil, save_dir, "input.png", crop_pcts)
    print(f"[{key}] wrote input.png  ({input_pil.size})"
          + (f" + crops {crop_pcts}" if crop_pcts else ""))

    # ── 2) Edited PNGs: resize to target_size to guarantee identical dims
    # (they're already 448x448 in this run, but be safe for future runs).
    base_path = os.path.join(out_root, "generated", "base", task_type, language,
                             f"{key}.png")
    dtr_path = os.path.join(out_root, "generated", "dtr", task_type, language,
                            f"{key}.png")
    if not os.path.exists(base_path):
        print(f"[{key}] FAIL: missing {base_path}")
        return False
    if not os.path.exists(dtr_path):
        print(f"[{key}] FAIL: missing {dtr_path}")
        return False
    base_pil = Image.open(base_path).convert("RGB")
    dtr_pil = Image.open(dtr_path).convert("RGB")
    if base_pil.size != target_size:
        base_pil = base_pil.resize(target_size, Image.LANCZOS)
    if dtr_pil.size != target_size:
        dtr_pil = dtr_pil.resize(target_size, Image.LANCZOS)
    _save_with_crops(base_pil, save_dir, "edited_base.png", crop_pcts)
    _save_with_crops(dtr_pil, save_dir, "edited_dtr.png", crop_pcts)
    print(f"[{key}] wrote edited_base.png, edited_dtr.png  ({target_size})")

    # ── 2b) Optional dtr2 (e.g. 3B) assets. Auto-probe same-root layout if
    # caller didn't pass --out_root_dtr2.
    dtr2_pil, dtr2_npz = _read_dtr2_assets(
        out_root_dtr2, out_root, key, task_type, language,
    )
    if dtr2_pil is not None:
        if dtr2_pil.size != target_size:
            dtr2_pil = dtr2_pil.resize(target_size, Image.LANCZOS)
        _save_with_crops(dtr2_pil, save_dir, "edited_dtr2.png", crop_pcts)
        print(f"[{key}] wrote edited_dtr2.png  ({target_size})")
        if dtr2_npz is None:
            src = out_root_dtr2 or f"{out_root}/attn_grids/{key}/daam_grids_dtr2.npz"
            print(f"[{key}] WARN: dtr2 attn npz missing ({src}); "
                  "dtr2 attention overlays will be skipped")
    elif out_root_dtr2:
        # Caller explicitly asked for sibling root but it's not there.
        print(f"[{key}] WARN: dtr2 edit png missing under {out_root_dtr2}; "
              "skipping dtr2 row")

    # ── 3) Attention overlays.
    keywords = _load_keywords(out_root, key, user_keywords)
    if not keywords:
        print(f"[{key}] WARN: no keywords resolved; skipping attention overlays")
        return True

    base_arr = np.asarray(base_pil).astype(np.float32) / 255.0
    dtr_arr = np.asarray(dtr_pil).astype(np.float32) / 255.0
    dtr2_arr: Optional[np.ndarray] = None
    if dtr2_pil is not None:
        dtr2_arr = np.asarray(dtr2_pil).astype(np.float32) / 255.0
    H, W = base_arr.shape[:2]

    base_npz = os.path.join(out_root, "attn_grids", key, "daam_grids_base.npz")
    dtr_npz = os.path.join(out_root, "attn_grids", key, "daam_grids_dtr.npz")
    if not (os.path.exists(base_npz) and os.path.exists(dtr_npz)):
        print(f"[{key}] WARN: base/dtr attn npz missing; skipping overlays")
        return True

    for kw in keywords:
        gb, sb = _resolve_grid_for_keyword(base_npz, kw, tokenizer)
        gd, sd = _resolve_grid_for_keyword(dtr_npz, kw, tokenizer)
        hb = _upsample_overlay(gb, (H, W)) if gb is not None else None
        hd = _upsample_overlay(gd, (H, W)) if gd is not None else None

        h2 = None
        s2 = "n/a"
        if dtr2_npz is not None and dtr2_arr is not None:
            g2, s2 = _resolve_grid_for_keyword(dtr2_npz, kw, tokenizer)
            h2 = _upsample_overlay(g2, (H, W)) if g2 is not None else None

        # Per-keyword shared vmin/vmax across all available rows so colours match.
        candidates = [hb, hd]
        if h2 is not None:
            candidates.append(h2)
        finite = [h for h in candidates if h is not None and h.sum() > 1e-12]
        if finite:
            vmin = float(min(h.min() for h in finite))
            vmax = float(max(h.max() for h in finite))
        else:
            vmin, vmax = 0.0, 1.0

        slug = _safe_name(kw)
        b_pil, ok_b = _overlay_pil(base_arr, hb, vmin, vmax, cmap, alpha)
        d_pil, ok_d = _overlay_pil(dtr_arr, hd, vmin, vmax, cmap, alpha)
        _save_with_crops(b_pil, save_dir, f"attn_{slug}_base.png", crop_pcts)
        _save_with_crops(d_pil, save_dir, f"attn_{slug}_dtr.png", crop_pcts)

        msg = (f"[{key}] kw='{kw}' base={sb} dtr={sd} -> "
               f"{'attn' if ok_b else 'plain'}_{slug}_base.png, "
               f"{'attn' if ok_d else 'plain'}_{slug}_dtr.png")
        if dtr2_arr is not None:
            d2_pil, ok_2 = _overlay_pil(dtr2_arr, h2, vmin, vmax, cmap, alpha)
            _save_with_crops(d2_pil, save_dir, f"attn_{slug}_dtr2.png", crop_pcts)
            msg += f", dtr2={s2} -> {'attn' if ok_2 else 'plain'}_{slug}_dtr2.png"
        print(msg)

    return True


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", required=True,
                        help="Pipeline output root (e.g. results/vis_sft_edit).")
    parser.add_argument("--keys", required=True,
                        help="Comma-separated keys to export, e.g. "
                             "sft_006649,sft_005235,sft_006811")
    parser.add_argument("--shard_root",
                        default="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/"
                                "unified_model/my_unilip/data/edit_sft",
                        help="Directory holding the SFT .tar shards.")
    parser.add_argument("--keywords", default="",
                        help="Optional comma-sep keyword override applied to "
                             "ALL --keys. Empty = use selected_keys.json.")
    parser.add_argument("--target_size", default="448,448",
                        help="W,H to resize input/edited to. Default 448,448 "
                             "matches pipeline_edit.py.")
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--tokenizer_path", default="",
                        help="Optional tokenizer path used by "
                             "_resolve_grid_for_keyword to recompute keyword "
                             "grids from raw attn (matches plot_sft_edit.py).")
    parser.add_argument("--out_root_dtr2", default="",
                        help="Optional sibling pipeline root holding the dtr2 "
                             "(e.g. UniLIP-3B) products. When set, exports an "
                             "extra edited_dtr2.png and attn_<kw>_dtr2.png per "
                             "keyword. The dtr2 pipeline must have been launched "
                             "with base=dtr=3B so the dtr/ branch holds 3B output.")
    parser.add_argument("--crop_pcts", default="",
                        help="Comma-separated extra crop versions, in percent "
                             "trimmed off EACH side. Example: '10,20' writes "
                             "an additional crop10/ and crop20/ subdir under "
                             "every key's png_export. Default: no extra crops.")
    args = parser.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if not keys:
        sys.exit("No --keys provided.")

    crop_pcts: List[int] = []
    if args.crop_pcts.strip():
        crop_pcts = [int(x) for x in args.crop_pcts.split(",") if x.strip()]
        for p in crop_pcts:
            if not (0 < p < 50):
                sys.exit(f"--crop_pcts entries must be in (0, 50); got {p}")
        print(f">>> Will also write crops at {crop_pcts}% (each side)")

    user_keywords: Optional[List[str]] = None
    if args.keywords.strip():
        user_keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    w, h = (int(x) for x in args.target_size.split(","))
    target_size = (w, h)

    tokenizer = None
    if args.tokenizer_path.strip():
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path, trust_remote_code=True
            )
            print(f">>> Loaded tokenizer from {args.tokenizer_path}")
        except Exception as e:  # noqa: BLE001
            print(f">>> WARN: tokenizer load failed ({e}); falling back to "
                  "legacy kw_* grids only.")

    shard_cache = _ShardCache(args.shard_root)
    n_ok = 0
    for k in keys:
        try:
            if export_one_key(
                out_root=args.out_root,
                key=k,
                shard_cache=shard_cache,
                user_keywords=user_keywords,
                target_size=target_size,
                cmap=args.cmap,
                alpha=args.alpha,
                tokenizer=tokenizer,
                out_root_dtr2=args.out_root_dtr2,
                crop_pcts=crop_pcts,
            ):
                n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[{k}] EXCEPTION: {e!r}")
            import traceback; traceback.print_exc()
    shard_cache.close()
    print(f"\nDone. {n_ok}/{len(keys)} keys fully exported under "
          f"{os.path.join(args.out_root, 'png_export')}/")


if __name__ == "__main__":
    main()

"""Plot Baseline-vs-DTR comparison figures for SFT-sampled edit cases.

Stage-3 sibling of ``plot_edit_compare.py`` for the SFT-sampled flow.
Differences vs the GEdit version:

  * No Qwen scoring => no ``score_diff.csv`` / ``edited_objects.json``.
  * Reads metadata from ``selected_keys.json`` (instruction, keyword,
    shard/stem) produced by ``visualize_edit_compare.py --source sft``.
  * Reads input image directly from the SFT .tar shard recorded in
    ``selected_keys.json`` (no HF dataset download).

Layout per case (2 rows x 3 cols):

    row 0 (Baseline)   [Input | Edit Result | "<keyword>" overlay]
    row 1 (DTR)        [Input | Edit Result | "<keyword>" overlay]

Outputs ``compare_figs/{key}/edit_compare.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from visualize_daam import _save_fig_dual, _upsample_overlay  # noqa: E402
from plot_edit_compare import (  # noqa: E402
    _resolve_grid_for_keyword, _draw_overlay,
)


# ────────────────────────────────────────────────────────────
# Metadata IO
# ────────────────────────────────────────────────────────────
def _load_selected_keys(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"selected_keys.json is not a list: {path}")
    return rows


def _load_keyword_map(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────
# Input-image lookup from SFT tar shards
# ────────────────────────────────────────────────────────────
class _ShardCache:
    """Open-once, find-many for SFT tar shards.

    For each shard we build {stem: TarInfo} of input.jpg members.
    """

    def __init__(self, shard_root: str):
        self.shard_root = shard_root
        self._tar_handles: Dict[str, tarfile.TarFile] = {}
        self._index: Dict[str, Dict[str, tarfile.TarInfo]] = {}

    def _ensure(self, shard_name: str) -> None:
        if shard_name in self._index:
            return
        path = os.path.join(self.shard_root, shard_name)
        if not os.path.exists(path):
            # Allow --shard_root to be a glob-like parent; try basename match.
            import glob
            cands = glob.glob(os.path.join(self.shard_root, "**", shard_name),
                              recursive=True)
            if cands:
                path = cands[0]
            else:
                raise FileNotFoundError(f"shard not found: {shard_name} "
                                        f"under {self.shard_root}")
        tf = tarfile.open(path, "r")
        self._tar_handles[shard_name] = tf
        idx: Dict[str, tarfile.TarInfo] = {}
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            stem, _, ext = name.partition(".")
            if ext.lower() == "input.jpg":
                idx[stem] = m
        self._index[shard_name] = idx

    def load_input(self, shard_name: str, stem: str) -> Optional[Image.Image]:
        try:
            self._ensure(shard_name)
        except Exception as e:
            print(f"   WARN: cannot open shard {shard_name}: {e}")
            return None
        m = self._index[shard_name].get(stem)
        if m is None:
            return None
        try:
            data = self._tar_handles[shard_name].extractfile(m).read()
            return Image.open(io.BytesIO(data)).convert("RGB").copy()
        except Exception as e:
            print(f"   WARN: cannot decode {shard_name}:{stem}: {e}")
            return None

    def close(self) -> None:
        for tf in self._tar_handles.values():
            try:
                tf.close()
            except Exception:
                pass


# ────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────
def plot_one(
    row: Dict[str, Any],
    out_root: str,
    shard_cache: _ShardCache,
    keyword_map: Dict[str, str],
    tokenizer,
    cmap: str = "jet",
    overlay_alpha: float = 0.55,
    baseline_label: str = "Baseline",
    dtr_label: str = "DTR (Ours)",
) -> Optional[str]:
    key = row["key"]
    keyword = (
        row.get("keyword")
        or keyword_map.get(key, "")
        or "(none)"
    ).strip() or "(none)"

    # 1) Input image from SFT tar.
    shard = row.get("_sft_shard")
    stem = row.get("_sft_stem")
    if not (shard and stem):
        print(f"[{key}] skip: no _sft_shard/_sft_stem in metadata.")
        return None
    input_pil = shard_cache.load_input(shard, stem)
    if input_pil is None:
        print(f"[{key}] skip: input image not loadable from {shard}:{stem}.")
        return None

    # 2) Generated images (path mirrors visualize_edit_compare.py).
    task_type = row.get("task_type", "sft_edit")
    language = row.get("instruction_language", "en")
    base_path = os.path.join(out_root, "generated", "base", task_type, language,
                             f"{key}.png")
    dtr_path = os.path.join(out_root, "generated", "dtr", task_type, language,
                            f"{key}.png")
    if not (os.path.exists(base_path) and os.path.exists(dtr_path)):
        print(f"[{key}] skip: missing generated image(s).")
        return None
    base_pil = Image.open(base_path).convert("RGB")
    dtr_pil = Image.open(dtr_path).convert("RGB")

    base_arr = np.asarray(base_pil).astype(np.float32) / 255.0
    dtr_arr = np.asarray(dtr_pil).astype(np.float32) / 255.0
    inp_arr = np.asarray(input_pil).astype(np.float32) / 255.0

    # 3) Heat-maps via the shared resolver.
    base_npz = os.path.join(out_root, "attn_grids", key, "daam_grids_base.npz")
    dtr_npz = os.path.join(out_root, "attn_grids", key, "daam_grids_dtr.npz")
    g_base, src_base = _resolve_grid_for_keyword(base_npz, keyword, tokenizer)
    g_dtr, src_dtr = _resolve_grid_for_keyword(dtr_npz, keyword, tokenizer)
    print(f"[{key}] keyword='{keyword}' base={src_base} dtr={src_dtr}")

    Hb, Wb = base_arr.shape[:2]
    Hd, Wd = dtr_arr.shape[:2]
    heat_b = _upsample_overlay(g_base, (Hb, Wb)) if g_base is not None else None
    heat_d = _upsample_overlay(g_dtr, (Hd, Wd)) if g_dtr is not None else None

    if heat_b is not None and heat_d is not None and (
            heat_b.sum() > 1e-12 or heat_d.sum() > 1e-12):
        vmin = float(min(heat_b.min(), heat_d.min()))
        vmax = float(max(heat_b.max(), heat_d.max()))
    else:
        vmin, vmax = 0.0, 1.0

    # 4) Figure.
    fig, axes = plt.subplots(
        2, 3,
        figsize=(3.4 * 3, 3.6 * 2),
        gridspec_kw={"wspace": 0.05, "hspace": 0.16},
    )

    # Col 0: input image (same on both rows).
    for r_idx, label in enumerate((baseline_label, dtr_label)):
        ax = axes[r_idx, 0]
        ax.imshow(inp_arr)
        ax.set_xticks([]); ax.set_yticks([])
        if r_idx == 0:
            ax.set_title("Input", fontsize=12)
        ax.set_ylabel(label, fontsize=12, fontweight="bold")

    # Col 1: edited result.
    axes[0, 1].imshow(base_arr); axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
    axes[0, 1].set_title("Edit Result", fontsize=12)
    axes[1, 1].imshow(dtr_arr); axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])

    # Col 2: heat-map overlay on the edited result.
    _draw_overlay(axes[0, 2], base_arr,
                  heat_b if (heat_b is not None and heat_b.sum() > 1e-12) else None,
                  keyword, vmin, vmax, cmap, overlay_alpha)
    _draw_overlay(axes[1, 2], dtr_arr,
                  heat_d if (heat_d is not None and heat_d.sum() > 1e-12) else None,
                  keyword, vmin, vmax, cmap, overlay_alpha)

    title = f"[{key}]  {row.get('instruction','')}"
    fig.suptitle(title, fontsize=10, y=0.995)

    save_dir = os.path.join(out_root, "compare_figs", key)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "edit_compare.pdf")
    _save_fig_dual(fig, save_path)
    plt.close(fig)
    return save_path


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SFT-sampled edit comparison figures (no Qwen scores)."
    )
    parser.add_argument("--out_root", required=True,
                        help="Same --out_root used by visualize_edit_compare.py "
                             "with --source sft.")
    parser.add_argument("--selected_keys", default="",
                        help="Default: {out}/selected_keys.json")
    parser.add_argument("--keywords_json", default="",
                        help="Default: {out}/keywords_auto.json (optional, "
                             "selected_keys already carries keywords).")
    parser.add_argument("--shard_root",
                        default="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/"
                                "unified_model/my_unilip/data/edit_sft",
                        help="Directory holding the SFT .tar shards.")
    parser.add_argument("--keys", default="",
                        help="Comma-separated subset of keys to plot. "
                             "Empty = all in selected_keys.json.")
    parser.add_argument("--top_k", type=int, default=0,
                        help="If >0, only plot the first K from the (filtered) "
                             "key list.")
    parser.add_argument("--tokenizer_path", default="",
                        help="HF model path for AutoTokenizer.from_pretrained "
                             "(used to recompute keyword grids from raw attn). "
                             "Default: $BASE_CKPT, else skip recompute.")
    parser.add_argument("--baseline_label", default="Baseline")
    parser.add_argument("--dtr_label", default="DTR (Ours)")
    args = parser.parse_args()

    out_root = os.path.abspath(args.out_root)
    sel_path = args.selected_keys or os.path.join(out_root, "selected_keys.json")
    kw_path = args.keywords_json or os.path.join(out_root, "keywords_auto.json")

    rows = _load_selected_keys(sel_path)
    keyword_map = _load_keyword_map(kw_path)
    by_key = {r["key"]: r for r in rows}

    # Resolve target keys.
    if args.keys.strip():
        wanted = [k.strip() for k in args.keys.split(",") if k.strip()]
    else:
        wanted = [r["key"] for r in rows]
    if args.top_k > 0:
        wanted = wanted[: args.top_k]
    print(f">>> Will plot {len(wanted)} keys.")

    # Optional tokenizer for recomputing keyword grids from raw attn.
    tokenizer = None
    tok_path = args.tokenizer_path or os.environ.get("BASE_CKPT", "")
    if tok_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tok_path, trust_remote_code=True,
            )
            print(f">>> Loaded tokenizer from {tok_path}")
        except Exception as e:
            print(f">>> WARN: failed to load tokenizer from {tok_path}: {e}")
            tokenizer = None
    else:
        print(">>> No --tokenizer_path / $BASE_CKPT; recompute-from-attn "
              "disabled, falling back to legacy kw_* grids only.")

    shard_cache = _ShardCache(args.shard_root)
    written: List[str] = []
    for key in wanted:
        r = by_key.get(key)
        if r is None:
            print(f"[{key}] not in selected_keys.json, skip.")
            continue
        path = plot_one(
            row=r, out_root=out_root,
            shard_cache=shard_cache,
            keyword_map=keyword_map,
            tokenizer=tokenizer,
            baseline_label=args.baseline_label,
            dtr_label=args.dtr_label,
        )
        if path:
            print(f"[{key}] -> {path}")
            written.append(path)

    shard_cache.close()
    print(f"\nDone. {len(written)}/{len(wanted)} figures written.")


if __name__ == "__main__":
    main()

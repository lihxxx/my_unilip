"""Plot 2x3 Baseline-vs-DTR edit-comparison figures.

Stage-3 output. For every selected key produces a 2-row x 3-col figure:

    row 0 (Baseline)   [Input | Edit Result | "<edited_object>" overlay]
    row 1 (DTR)        [Input | Edit Result | "<edited_object>" overlay]

Reads:
  * input_image_raw     -> from the HF dataset (cached locally)
  * generated/<tag>/.../{key}.png         -> from visualize_edit_compare.py
  * attn_grids/{key}/daam_grids_{tag}.npz -> from visualize_edit_compare.py
                                             with --capture_attn
  * scores/score_diff.csv                 -> for the score footer & defaults
  * scores/edited_objects.json            -> for the keyword

If a key has NO npz file (e.g. user only ran the no-capture pass), the
heat-map column is replaced by a "no attention captured" placeholder so the
figure still renders.

Outputs ``compare_figs/{key}/edit_compare.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from visualize_daam import _normalize, _save_fig_dual, _upsample_overlay  # noqa: E402


# ────────────────────────────────────────────────────────────
# IO helpers
# ────────────────────────────────────────────────────────────
def _load_csv(csv_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("base_SC", "dtr_SC", "base_PQ", "dtr_PQ",
                      "base_O", "dtr_O", "delta_SC", "delta_PQ", "delta_O"):
                if r.get(k) not in (None, ""):
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def _load_grid_npz(path: str) -> Dict[str, np.ndarray]:
    """Return {keyword_str: grid}. The file stores ``kw_{i}_{name}`` keys
    plus _meta_* arrays which we ignore."""
    if not os.path.exists(path):
        return {}
    npz = np.load(path)
    out: Dict[str, np.ndarray] = {}
    for k in npz.files:
        if k.startswith("_meta"):
            continue
        if not k.startswith("kw_"):
            continue
        # 'kw_{i}_{name}' -> name (name itself may contain underscores).
        rest = k[len("kw_"):]
        if "_" in rest:
            _, name = rest.split("_", 1)
        else:
            name = rest
        out[name] = np.asarray(npz[k])
    return out


def _pick_grid_for_keyword(
    grids: Dict[str, np.ndarray],
    keyword: str,
) -> Optional[np.ndarray]:
    """Best-effort lookup: exact match first, then case-insensitive, then any."""
    if not grids:
        return None
    if keyword in grids:
        return grids[keyword]
    low = {k.lower(): v for k, v in grids.items()}
    if keyword.lower() in low:
        return low[keyword.lower()]
    # Fallback: take the first non-zero grid (often there's only one).
    for v in grids.values():
        if v.sum() > 1e-12:
            return v
    return next(iter(grids.values()))


def _load_input_image(out_root: str, key: str, dataset_name: str,
                       dataset_split: str, language: str,
                       cache: Dict[str, Image.Image]) -> Optional[Image.Image]:
    """Load input_image_raw lazily from the HF dataset."""
    if "_ds" not in cache:
        from datasets import load_dataset
        ds = load_dataset(dataset_name)[dataset_split]
        cache["_ds"] = ds
        cache["_by_key"] = {it["key"]: it for it in ds
                            if language == "all"
                            or it["instruction_language"] == language}
    by_key = cache["_by_key"]
    it = by_key.get(key)
    if it is None:
        return None
    return it["input_image_raw"].convert("RGB")


# ────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────
def _draw_overlay(ax, base_img: np.ndarray, heat: Optional[np.ndarray],
                   keyword: str, vmin: float, vmax: float,
                   cmap: str, alpha: float,
                   placeholder: str = "(no attention captured)") -> None:
    ax.imshow(base_img)
    ax.set_xticks([]); ax.set_yticks([])
    if heat is None:
        ax.set_title(f"\"{keyword}\"\n{placeholder}", fontsize=10, color="gray")
        return
    denom = max(vmax - vmin, 1e-8)
    heat_n = (heat - vmin) / denom
    ax.imshow(heat_n, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_title(f"\"{keyword}\"", fontsize=12)


def _draw_score_footer(ax, label: str, sc: float, pq: float, ov: float,
                        delta_o: Optional[float] = None) -> None:
    parts = [f"SC={sc:.2f}", f"PQ={pq:.2f}", f"O={ov:.2f}"]
    if delta_o is not None:
        sign = "+" if delta_o >= 0 else ""
        parts.append(f"ΔO={sign}{delta_o:.2f}")
    txt = " | ".join(parts)
    ax.text(0.99, 0.02, f"{label}: {txt}",
             transform=ax.transAxes, ha="right", va="bottom",
             fontsize=9, color="white",
             bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))


def plot_one(
    row: Dict[str, Any],
    out_root: str,
    save_dir: str,
    ds_cache: Dict[str, Any],
    dataset_name: str,
    dataset_split: str,
    cmap: str = "jet",
    overlay_alpha: float = 0.55,
    baseline_label: str = "Baseline",
    dtr_label: str = "DTR (Ours)",
) -> Optional[str]:
    key = row["key"]
    keyword = (row.get("edited_object") or "").strip() or "(none)"

    # 1) Inputs.
    input_pil = _load_input_image(
        out_root, key, dataset_name, dataset_split,
        row.get("instruction_language", "en"), ds_cache,
    )
    if input_pil is None:
        print(f"[{key}] skip: input image not found in dataset.")
        return None

    base_path = row["base_image_path"]
    dtr_path = row["dtr_image_path"]
    if not (os.path.exists(base_path) and os.path.exists(dtr_path)):
        print(f"[{key}] skip: missing generated image(s).")
        return None
    base_pil = Image.open(base_path).convert("RGB")
    dtr_pil = Image.open(dtr_path).convert("RGB")

    base_arr = np.asarray(base_pil).astype(np.float32) / 255.0
    dtr_arr = np.asarray(dtr_pil).astype(np.float32) / 255.0
    inp_arr = np.asarray(input_pil).astype(np.float32) / 255.0

    # 2) Heat-maps (may be missing).
    base_grids = _load_grid_npz(
        os.path.join(out_root, "attn_grids", key, "daam_grids_base.npz"))
    dtr_grids = _load_grid_npz(
        os.path.join(out_root, "attn_grids", key, "daam_grids_dtr.npz"))
    g_base = _pick_grid_for_keyword(base_grids, keyword)
    g_dtr = _pick_grid_for_keyword(dtr_grids, keyword)

    Hb, Wb = base_arr.shape[:2]
    Hd, Wd = dtr_arr.shape[:2]
    heat_b = _upsample_overlay(g_base, (Hb, Wb)) if g_base is not None else None
    heat_d = _upsample_overlay(g_dtr, (Hd, Wd)) if g_dtr is not None else None

    if heat_b is not None and heat_d is not None and (heat_b.sum() > 1e-12 or heat_d.sum() > 1e-12):
        vmin = float(min(heat_b.min(), heat_d.min()))
        vmax = float(max(heat_b.max(), heat_d.max()))
    else:
        vmin, vmax = 0.0, 1.0

    # 3) Figure.
    fig, axes = plt.subplots(
        2, 3,
        figsize=(3.4 * 3, 3.6 * 2),
        gridspec_kw={"wspace": 0.05, "hspace": 0.16},
    )

    # Col 0: input image (same on both rows for clarity).
    for r_idx, (label, arr) in enumerate(((baseline_label, base_arr),
                                          (dtr_label, dtr_arr))):
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

    # Score footer on the edited-result column (most informative spot).
    _draw_score_footer(
        axes[0, 1], baseline_label,
        row["base_SC"], row["base_PQ"], row["base_O"], delta_o=None,
    )
    _draw_score_footer(
        axes[1, 1], dtr_label,
        row["dtr_SC"], row["dtr_PQ"], row["dtr_O"],
        delta_o=row.get("delta_O"),
    )

    # Col 2: heat-map overlay on the edited result.
    _draw_overlay(axes[0, 2], base_arr,
                   heat_b if (heat_b is not None and heat_b.sum() > 1e-12) else None,
                   keyword, vmin, vmax, cmap, overlay_alpha)
    _draw_overlay(axes[1, 2], dtr_arr,
                   heat_d if (heat_d is not None and heat_d.sum() > 1e-12) else None,
                   keyword, vmin, vmax, cmap, overlay_alpha)

    title = (f"[{row.get('task_type','')}|{row.get('instruction_language','')}|{key}]  "
             f"{row.get('instruction','')}")
    fig.suptitle(title, fontsize=10, y=0.995)

    os.makedirs(save_dir, exist_ok=True)
    save_base = os.path.join(save_dir, "edit_compare.pdf")
    _save_fig_dual(fig, save_base)
    plt.close(fig)
    return save_base


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 2x3 Baseline-vs-DTR edit comparison figures.",
    )
    parser.add_argument("--out_root", required=True,
                        help="Same --out_root used by stages 1/2.")
    parser.add_argument("--csv", default="",
                        help="score_diff.csv path. Default: {out}/scores/score_diff.csv")
    parser.add_argument("--keys", default="",
                        help="Comma-separated key list to plot. "
                             "Overrides --topk_json / --top_k.")
    parser.add_argument("--topk_json", default="",
                        help="JSON produced by stage 2 with .keys list. "
                             "Default: {out}/scores/selected_topk.json")
    parser.add_argument("--top_k", type=int, default=8,
                        help="If neither --keys nor --topk_json is given, "
                             "take this many top-delta_O rows from the CSV.")
    parser.add_argument("--dataset_name", default="stepfun-ai/GEdit-Bench")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--baseline_label", default="Baseline")
    parser.add_argument("--dtr_label", default="DTR (Ours)")
    args = parser.parse_args()

    out_root = os.path.abspath(args.out_root)
    csv_path = args.csv or os.path.join(out_root, "scores", "score_diff.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows = _load_csv(csv_path)
    by_key = {r["key"]: r for r in rows}

    # Resolve target keys.
    if args.keys.strip():
        wanted = [k.strip() for k in args.keys.split(",") if k.strip()]
    elif args.topk_json or os.path.exists(
            os.path.join(out_root, "scores", "selected_topk.json")):
        path = args.topk_json or os.path.join(out_root, "scores", "selected_topk.json")
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        wanted = blob.get("keys", [])[: args.top_k] if args.top_k > 0 else blob.get("keys", [])
    else:
        wanted = [r["key"] for r in rows[: args.top_k]]
    print(f">>> Will plot {len(wanted)} keys.")

    ds_cache: Dict[str, Any] = {}
    out_fig_root = os.path.join(out_root, "compare_figs")
    written: List[str] = []
    for key in wanted:
        r = by_key.get(key)
        if r is None:
            print(f"[{key}] not in CSV, skip.")
            continue
        save_dir = os.path.join(out_fig_root, key)
        path = plot_one(
            row=r, out_root=out_root, save_dir=save_dir,
            ds_cache=ds_cache,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            baseline_label=args.baseline_label,
            dtr_label=args.dtr_label,
        )
        if path:
            print(f"[{key}] -> {path}")
            written.append(path)

    print(f"\nDone. {len(written)}/{len(wanted)} figures written.")


if __name__ == "__main__":
    main()

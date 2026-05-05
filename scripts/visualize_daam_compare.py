"""DAAM-style cross-attention visualisation: DTR vs Baseline side-by-side.

For every prompt:
  1. Load Baseline ckpt, generate image with the given seed, capture Sana DiT
     cross-attention -> compute keyword grids -> free model.
  2. Load DTR ckpt, repeat with the SAME seed.
  3. Plot a 2-row x (1 + n_keywords)-col figure:

         row 0  [Baseline image]  [kw1 overlay]  [kw2 overlay]  [kw3 overlay]
         row 1  [DTR      image]  [kw1 overlay]  [kw2 overlay]  [kw3 overlay]

We deliberately reload the full model in between passes (instead of holding
both in GPU memory at once) so this works on a single A100/H100 even when
both checkpoints are 7B-class. It is slower but safer.

All heavy lifting (monkey-patch, capture, aggregation, grid extraction) is
imported from ``scripts/visualize_daam.py`` so we keep a single source of
truth for the cross-attn logic.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sibling script importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from visualize_daam import (  # noqa: E402  (project sibling import)
    DiTCrossAttnContext,
    _save_fig_dual,
    _upsample_overlay,
    _normalize,
    aggregate_cross_attention,
    build_prompts,
    find_keyword_token_positions,
    keyword_grid,
    set_seed,
)
from unilip.constants import *  # noqa: F401,F403,E402
from unilip.model.builder import load_pretrained_model_general  # noqa: E402
from unilip.utils import disable_torch_init  # noqa: E402
from unilip.mm_utils import get_model_name_from_path  # noqa: E402
from unilip.pipeline_gen import CustomGenPipeline  # noqa: E402


# ────────────────────────────────────────────────────────────
# Per-pass result container
# ────────────────────────────────────────────────────────────
@dataclass
class PassResult:
    image: Image.Image
    keyword_grids: Dict[str, np.ndarray]
    n_steps: int
    n_layers: int


def run_one_pass(
    model,
    tokenizer,
    pipe: CustomGenPipeline,
    prompt_text: str,
    keywords: List[str],
    seed: int,
    guidance_scale: float,
    step_window: Tuple[float, float],
    layer_indices: Optional[List[int]],
    grid_side_hint: int = 16,
) -> PassResult:
    """Generate one image and return DAAM keyword grids for the given prompt."""
    prompts = build_prompts(prompt_text)
    set_seed(seed)
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)

    with DiTCrossAttnContext(model) as ctx:
        image = pipe(prompts, guidance_scale=guidance_scale, generator=gen)

    n_steps = len(ctx.capture.attn)
    n_layers = len(ctx.capture.attn[0]) if n_steps else 0
    if n_steps == 0:
        return PassResult(image=image, keyword_grids={}, n_steps=0, n_layers=0)

    s0 = max(0, int(round(step_window[0] * n_steps)))
    s1 = max(s0 + 1, int(round(step_window[1] * n_steps)))
    step_indices = list(range(s0, s1))

    attn_qs = aggregate_cross_attention(ctx.capture, step_indices, layer_indices)
    Q, S = attn_qs.shape
    grid_side = int(round(math.sqrt(Q)))
    if grid_side * grid_side != Q:
        attn_qs = attn_qs[: grid_side * grid_side]
        grid_side = int(round(math.sqrt(attn_qs.shape[0])))

    pos_prompt = prompts[0]
    grids: Dict[str, np.ndarray] = {}
    for kw in keywords:
        positions = find_keyword_token_positions(tokenizer, pos_prompt, kw)
        positions = [p for p in positions if 0 <= p < S]
        if not positions:
            grids[kw] = np.zeros((grid_side, grid_side), dtype=np.float32)
            print(f"   keyword '{kw}': NOT FOUND in prompt tokens")
            continue
        grids[kw] = keyword_grid(attn_qs, positions, grid_size=grid_side)
        print(f"   keyword '{kw}': positions={positions}, "
              f"max={grids[kw].max():.4f}, mean={grids[kw].mean():.4f}")

    return PassResult(
        image=image,
        keyword_grids=grids,
        n_steps=n_steps,
        n_layers=n_layers,
    )


# ────────────────────────────────────────────────────────────
# Side-by-side plot
# ────────────────────────────────────────────────────────────
def plot_compare_rows(
    baseline: PassResult,
    dtr: PassResult,
    keywords: List[str],
    save_path: str,
    cmap: str = "jet",
    overlay_alpha: float = 0.55,
    title: Optional[str] = None,
    row_labels: Tuple[str, str] = ("Baseline", "DTR"),
    shared_norm_per_keyword: bool = True,
) -> None:
    """Save 2-row x (1+K)-col DAAM comparison figure as ``save_path.{pdf,png}``.

    When ``shared_norm_per_keyword`` is True (default) the same vmin/vmax is
    used for both rows of a given keyword, so the visual intensity is directly
    comparable between Baseline and DTR.
    """
    n_panels = 1 + len(keywords)

    base_img = np.asarray(baseline.image.convert("RGB")).astype(np.float32) / 255.0
    dtr_img = np.asarray(dtr.image.convert("RGB")).astype(np.float32) / 255.0
    H_b, W_b = base_img.shape[:2]
    H_d, W_d = dtr_img.shape[:2]

    fig, axes = plt.subplots(
        2, n_panels,
        figsize=(3.2 * n_panels, 3.4 * 2),
        gridspec_kw={"wspace": 0.05, "hspace": 0.12},
    )
    if n_panels == 1:
        axes = np.array(axes).reshape(2, 1)

    # Column 0: generated images.
    axes[0, 0].imshow(base_img)
    axes[0, 0].set_title("Generated", fontsize=12)
    axes[0, 0].set_ylabel(row_labels[0], fontsize=13, fontweight="bold")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    axes[1, 0].imshow(dtr_img)
    axes[1, 0].set_ylabel(row_labels[1], fontsize=13, fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # Columns 1..K: keyword overlays.
    for col, kw in enumerate(keywords, start=1):
        g_base = baseline.keyword_grids.get(kw)
        g_dtr = dtr.keyword_grids.get(kw)

        # Per-keyword shared normalisation across the two rows.
        if shared_norm_per_keyword and g_base is not None and g_dtr is not None \
                and (g_base.sum() > 1e-12 or g_dtr.sum() > 1e-12):
            heat_b = _upsample_overlay(g_base, (H_b, W_b))
            heat_d = _upsample_overlay(g_dtr, (H_d, W_d))
            vmin = min(heat_b.min(), heat_d.min())
            vmax = max(heat_b.max(), heat_d.max())
            denom = max(vmax - vmin, 1e-8)
            heat_b = (heat_b - vmin) / denom
            heat_d = (heat_d - vmin) / denom
        else:
            heat_b = _normalize(_upsample_overlay(g_base, (H_b, W_b))) \
                if g_base is not None and g_base.sum() > 1e-12 else None
            heat_d = _normalize(_upsample_overlay(g_dtr, (H_d, W_d))) \
                if g_dtr is not None and g_dtr.sum() > 1e-12 else None

        # Row 0 (baseline)
        ax_b = axes[0, col]
        ax_b.imshow(base_img)
        if heat_b is None:
            ax_b.set_title(f"\"{kw}\"\n(no token match)", fontsize=11, color="red")
        else:
            ax_b.imshow(heat_b, cmap=cmap, alpha=overlay_alpha, vmin=0.0, vmax=1.0)
            ax_b.set_title(f"\"{kw}\"", fontsize=12)
        ax_b.set_xticks([]); ax_b.set_yticks([])

        # Row 1 (DTR)
        ax_d = axes[1, col]
        ax_d.imshow(dtr_img)
        if heat_d is None:
            ax_d.set_title("(no token match)", fontsize=10, color="red")
        else:
            ax_d.imshow(heat_d, cmap=cmap, alpha=overlay_alpha, vmin=0.0, vmax=1.0)
        ax_d.set_xticks([]); ax_d.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=11, y=0.995)

    _save_fig_dual(fig, save_path)
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def _free_model(model):
    """Best-effort GPU memory release between the two passes."""
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_pass_for_all_prompts(
    model_path: str,
    tag: str,
    jobs: List[Tuple[str, str, List[str], int]],
    guidance_scale: float,
    step_window: Tuple[float, float],
    layer_indices: Optional[List[int]],
    output_dir: str,
) -> Dict[str, PassResult]:
    """Load model, run all prompts, save per-prompt single-row figures, return
    dict ``{pid: PassResult}``."""
    disable_torch_init()
    mp = os.path.expanduser(model_path)
    mname = get_model_name_from_path(mp)
    print(f"\n>>> Loading {tag.upper()} model: {mp}")
    tok, mdl, _ = load_pretrained_model_general(
        "UniLIP_InternVLForCausalLM", mp, None, mname
    )
    mdl.eval()
    pipe = CustomGenPipeline(multimodal_encoder=mdl, tokenizer=tok)
    print(f">>> {tag.upper()} model loaded.")

    results: Dict[str, PassResult] = {}
    for pid, ptxt, kws, seed in tqdm(jobs, desc=f"DAAM[{tag}]"):
        out_dir = os.path.join(output_dir, pid)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{tag}|{pid}] {ptxt[:80]}{'…' if len(ptxt) > 80 else ''}")
        try:
            res = run_one_pass(
                model=mdl, tokenizer=tok, pipe=pipe,
                prompt_text=ptxt, keywords=kws, seed=seed,
                guidance_scale=guidance_scale,
                step_window=step_window,
                layer_indices=layer_indices,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}|{pid}] ERROR: {e!r}")
            import traceback; traceback.print_exc()
            continue

        # Save raw image and grids (so we can re-plot without re-running).
        res.image.save(os.path.join(out_dir, f"generated_{tag}.png"))
        np.savez(
            os.path.join(out_dir, f"daam_grids_{tag}.npz"),
            **{f"kw_{i}_{kw}": g for i, (kw, g) in enumerate(res.keyword_grids.items())},
        )
        results[pid] = res

    _free_model(mdl)
    del pipe, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="DAAM-style cross-attention: DTR vs Baseline side-by-side",
    )
    parser.add_argument("--baseline_model_path", required=True)
    parser.add_argument("--dtr_model_path", required=True)
    parser.add_argument("--prompt_json", required=True)
    parser.add_argument("--output_dir", default="results/vis_daam_compare")
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument(
        "--step_window", default="0.4,0.6",
        help="Fractional window of denoising steps to average over.",
    )
    parser.add_argument(
        "--layers", default="",
        help="Comma-separated DiT layer indices to average. Empty = all layers.",
    )
    parser.add_argument(
        "--baseline_label", default="Baseline",
        help="Row label for the baseline pass.",
    )
    parser.add_argument(
        "--dtr_label", default="DTR (Ours)",
        help="Row label for the DTR pass.",
    )
    args = parser.parse_args()

    s_lo, s_hi = (float(x) for x in args.step_window.split(","))
    step_window = (s_lo, s_hi)

    layer_indices: Optional[List[int]] = None
    if args.layers.strip():
        layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    # Load prompts
    with open(args.prompt_json, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    jobs: List[Tuple[str, str, List[str], int]] = []
    for item in prompt_data:
        pid = str(item["id"])
        ptxt = item["prompt"]
        kws = [k.strip() for k in item.get("keywords", "").split(",") if k.strip()]
        item_seed = item.get("seed", args.seed)
        jobs.append((pid, ptxt, kws, item_seed))
        if args.max_prompts is not None and len(jobs) >= args.max_prompts:
            break
    print(f"Loaded {len(jobs)} prompts from {args.prompt_json}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Pass 1: Baseline
    base_results = _run_pass_for_all_prompts(
        model_path=args.baseline_model_path,
        tag="base",
        jobs=jobs,
        guidance_scale=args.guidance_scale,
        step_window=step_window,
        layer_indices=layer_indices,
        output_dir=args.output_dir,
    )

    # Pass 2: DTR
    dtr_results = _run_pass_for_all_prompts(
        model_path=args.dtr_model_path,
        tag="dtr",
        jobs=jobs,
        guidance_scale=args.guidance_scale,
        step_window=step_window,
        layer_indices=layer_indices,
        output_dir=args.output_dir,
    )

    # Plot comparison figures
    print("\n>>> Plotting side-by-side comparison figures …")
    for pid, ptxt, kws, _ in jobs:
        if pid not in base_results or pid not in dtr_results:
            print(f"[{pid}] missing one of the passes, skip.")
            continue
        out_dir = os.path.join(args.output_dir, pid)
        plot_compare_rows(
            baseline=base_results[pid],
            dtr=dtr_results[pid],
            keywords=kws,
            save_path=os.path.join(out_dir, "daam_compare.pdf"),
            title=f"[{pid}] {ptxt}",
            row_labels=(args.baseline_label, args.dtr_label),
        )
        print(f"[{pid}] -> {os.path.join(out_dir, 'daam_compare.pdf')}")

    print("\nAll done.")


if __name__ == "__main__":
    main()

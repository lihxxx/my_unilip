#!/usr/bin/env python
"""
visualize_dtr.py – Visualize Dynamic Token Routing (DTR) weights and
text-object correspondence in UniLIP image generation.

Supports two modes:
  1. Single-prompt mode  (--prompt / --keywords)
  2. Batch mode          (--prompt_json  reads a JSON file)

Per-prompt outputs (saved under <output_dir>/<id>/):
    generated.png         – the generated image
    dtr_heatmaps.pdf      – per-layer routing weight heatmaps + dominant-layer map
    object_regions.pdf    – per-keyword similarity heatmap + combined colour overlay
    combined.pdf          – single two-row figure with both DTR & object regions

Examples
--------
    # Single prompt
    python scripts/visualize_dtr.py \
        --model_path results/your_checkpoint \
        --prompt "A red cat sitting on a blue sofa in a cozy room" \
        --keywords "cat,sofa,room" \
        --output_dir results/vis_dtr

    # Batch from JSON  (same format as gen_image.py's prompts.json,
    #                    with an optional "keywords" field per item)
    python scripts/visualize_dtr.py \
        --model_path results/your_checkpoint \
        --prompt_json prompts_vis.json \
        --output_dir results/vis_dtr

    prompts_vis.json example:
    [
        {"id": "001", "prompt": "A red cat on a blue sofa", "keywords": "cat,sofa"},
        {"id": "002", "prompt": "A castle on a cliff at sunset", "keywords": "castle,cliff,sunset"}
    ]
"""

import os
import sys
import json
import random
import argparse
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from unilip.constants import *
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init
from unilip.mm_utils import get_model_name_from_path
from unilip.pipeline_gen import CustomGenPipeline

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.titlesize": 15,
    "savefig.facecolor": "white",
})

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


# ────────────────────────────────────────────────────────────
# Hook-based capture context
# ────────────────────────────────────────────────────────────
class DTRCaptureContext:
    """Context manager that monkey-patches the DynamicTokenRouter to capture
    routing weights and fused hidden states during a single generation call.

    Also places a forward hook on the LLM backbone so that text-visual
    similarity can be computed even when DTR is disabled.
    """

    def __init__(self, model):
        self.model = model
        self.routing_weights: Optional[torch.Tensor] = None  # [B, N, K]
        self.fused_hidden: Optional[torch.Tensor] = None     # [B, N, D]
        self.last_hidden: Optional[torch.Tensor] = None      # [B, N, D]

        self._has_router = (
            getattr(model.model.config, "enable_dynamic_routing", False)
            and hasattr(model.model, "dynamic_router")
        )
        self._orig_fwd = None
        self._llm_hook_handle = None

    # ── enter / exit ──────────────────────────────────────
    def __enter__(self):
        # Hook 1: replace DynamicTokenRouter.forward
        if self._has_router:
            router = self.model.model.dynamic_router
            self._orig_fwd = router.forward
            ctx = self

            def _capturing_forward(selected_hiddens):
                stacked = torch.stack(selected_hiddens, dim=-2)
                scaled = stacked * router.layer_scales.view(1, 1, -1, 1)
                query = selected_hiddens[-1]
                logits = router.router(query) / router.temperature
                weights = F.softmax(logits, dim=-1)

                ctx.routing_weights = weights.detach().cpu()

                fused = (scaled * weights.unsqueeze(-1)).sum(dim=-2)
                result = router.proj(router.out_norm(fused))
                ctx.fused_hidden = result.detach().cpu()
                return result

            router.forward = _capturing_forward

        # Hook 2: LLM output → last hidden state (fallback for non-DTR models)
        ctx_ref = self

        def _llm_hook(module, input, output):
            hs = getattr(output, "hidden_states", None)
            if hs is None and isinstance(output, tuple):
                for item in output:
                    if isinstance(item, (tuple, list)) and len(item) > 0 and isinstance(item[0], torch.Tensor):
                        hs = item
                        break
            if hs is not None and len(hs) > 0:
                ctx_ref.last_hidden = hs[-1].detach().cpu()

        self._llm_hook_handle = self.model.model.language_model.register_forward_hook(_llm_hook)
        return self

    def __exit__(self, *exc):
        if self._has_router and self._orig_fwd is not None:
            self.model.model.dynamic_router.forward = self._orig_fwd
        if self._llm_hook_handle is not None:
            self._llm_hook_handle.remove()

    @property
    def hidden_for_similarity(self) -> Optional[torch.Tensor]:
        """Best available hidden states: routed > last-layer."""
        return self.fused_hidden if self.fused_hidden is not None else self.last_hidden


# ────────────────────────────────────────────────────────────
# Token utilities
# ────────────────────────────────────────────────────────────
def find_keyword_token_positions(tokenizer, text: str, keyword: str) -> List[int]:
    """Return token indices of *keyword* inside the tokenised *text*.

    Uses an exact-subsequence match first, then falls back to per-token
    fuzzy matching so that sub-word tokenisations are still captured.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    kw_ids = tokenizer.encode(keyword, add_special_tokens=False)

    # exact subsequence
    kw_len = len(kw_ids)
    for i in range(len(ids) - kw_len + 1):
        if ids[i : i + kw_len] == kw_ids:
            return list(range(i, i + kw_len))

    # fuzzy: decode each token and check overlap
    kw_low = keyword.lower()
    positions = []
    for i, tid in enumerate(ids):
        tok_str = tokenizer.decode([tid]).strip().lower()
        if tok_str and (kw_low in tok_str or tok_str in kw_low):
            positions.append(i)
    return positions


# ────────────────────────────────────────────────────────────
# Prompt / seed helpers
# ────────────────────────────────────────────────────────────
def build_prompts(prompt_text: str):
    tpl = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n<img>"
    return [
        tpl.format(input=f"Generate an image: {prompt_text}"),
        tpl.format(input="Generate an image."),
    ]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────
def _single_color_cmap(hex_color: str):
    """Transparent → *hex_color* linear colormap."""
    rgba = mcolors.to_rgba(hex_color)
    return mcolors.LinearSegmentedColormap.from_list("c", [(1, 1, 1, 0), rgba], N=256)


def _upsample_grid(tensor_1d: torch.Tensor, n_query: int, H: int, W: int) -> np.ndarray:
    """Reshape a [N_query] tensor to [H, W] via bilinear upsampling."""
    gs = int(np.sqrt(n_query))
    grid = tensor_1d.view(gs, gs).float()[None, None]
    return F.interpolate(grid, (H, W), mode="bilinear", align_corners=False).squeeze().numpy()


def _compute_keyword_similarity_maps(
    hidden_states: torch.Tensor,
    tokenizer,
    pos_prompt: str,
    keywords: List[str],
    n_query: int,
    image_hw: tuple,
) -> Dict[str, np.ndarray]:
    """Cosine similarity between each keyword's mean hidden state and every
    visual token, upsampled to the original image resolution.

    Returns {keyword: ndarray[H, W] in [0, 1]}.
    """
    hs = hidden_states[0]  # take positive prompt (idx 0)
    n_text = hs.shape[0] - n_query
    text_h = hs[:n_text]
    vis_h = hs[n_text:]
    H, W = image_hw

    maps: Dict[str, np.ndarray] = {}
    for kw in keywords:
        pos = find_keyword_token_positions(tokenizer, pos_prompt, kw)
        if not pos:
            print(f"  Warning: keyword '{kw}' not found in prompt tokens, skipping")
            continue
        kw_vec = F.normalize(text_h[pos].float().mean(0, keepdim=True), dim=-1)
        vis_norm = F.normalize(vis_h.float(), dim=-1)
        sim = (vis_norm @ kw_vec.T).squeeze(-1)
        sim_up = _upsample_grid(sim, n_query, H, W)
        sim_up = (sim_up - sim_up.min()) / (sim_up.max() - sim_up.min() + 1e-8)
        maps[kw] = sim_up
    return maps


# ────────────────────────────────────────────────────────────
# Plotting functions
# ────────────────────────────────────────────────────────────
def plot_dtr_heatmaps(
    image: Image.Image,
    routing_weights: torch.Tensor,
    n_query: int,
    layer_indices: Optional[List[int]] = None,
    save_path: str = "dtr_heatmaps.pdf",
):
    """Per-layer routing-weight heatmaps + dominant-layer + entropy maps."""
    vis_w = routing_weights[0, -n_query:]  # [N_query, K]
    K = vis_w.shape[1]
    gs = int(np.sqrt(n_query))
    img_np = np.array(image)
    H, W = img_np.shape[:2]

    grid = vis_w.view(gs, gs, K).permute(2, 0, 1).unsqueeze(0).float()
    up = F.interpolate(grid, (H, W), mode="bilinear", align_corners=False).squeeze(0).numpy()

    names = (
        [f"Layer {idx}" for idx in layer_indices]
        if layer_indices
        else [f"Layer {i + 1}" for i in range(K)]
    )

    ncols = K + 3  # original + K layers + dominant + entropy
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.5))

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Generated Image", weight="bold")
    axes[0].axis("off")

    # Per-layer heatmaps
    for k in range(K):
        w = up[k]
        wn = (w - w.min()) / (w.max() - w.min() + 1e-8)
        ax = axes[k + 1]
        ax.imshow(img_np, alpha=0.35)
        im = ax.imshow(wn, cmap="jet", alpha=0.65, vmin=0, vmax=1)
        ax.set_title(names[k])
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Dominant layer map
    dom = vis_w.argmax(-1).view(gs, gs).float()
    dom_up = F.interpolate(dom[None, None], (H, W), mode="nearest").squeeze().numpy()
    ax = axes[K + 1]
    ax.imshow(img_np, alpha=0.35)
    cmap_d = plt.cm.get_cmap("Set1", K)
    ax.imshow(dom_up, cmap=cmap_d, alpha=0.6, vmin=-0.5, vmax=K - 0.5)
    ax.set_title("Dominant Layer", weight="bold")
    ax.axis("off")
    ax.legend(
        handles=[mpatches.Patch(color=cmap_d(i), label=names[i]) for i in range(K)],
        loc="lower right", fontsize=7, framealpha=0.8,
    )

    # Routing entropy map: H = -sum(p log p)
    eps = 1e-8
    entropy = -(vis_w * (vis_w + eps).log()).sum(-1)  # [N_query]
    ent_up = _upsample_grid(entropy, n_query, H, W)
    ent_up = (ent_up - ent_up.min()) / (ent_up.max() - ent_up.min() + 1e-8)
    ax = axes[K + 2]
    ax.imshow(img_np, alpha=0.35)
    im = ax.imshow(ent_up, cmap="magma", alpha=0.65)
    ax.set_title("Routing Entropy", weight="bold")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Dynamic Token Routing – Layer Preference", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"DTR heatmaps → {save_path}")
    plt.close(fig)


def plot_object_regions(
    image: Image.Image,
    hidden_states: torch.Tensor,
    tokenizer,
    pos_prompt: str,
    keywords: List[str],
    n_query: int,
    save_path: str = "object_regions.pdf",
):
    """Per-keyword cosine-similarity heatmap + combined colour overlay."""
    img_np = np.array(image)
    H, W = img_np.shape[:2]

    sim_maps = _compute_keyword_similarity_maps(
        hidden_states, tokenizer, pos_prompt, keywords, n_query, (H, W)
    )
    valid_kws = list(sim_maps.keys())
    if not valid_kws:
        print("No valid keywords → skipping object region visualisation")
        return

    nk = len(valid_kws)
    fig, axes = plt.subplots(1, nk + 2, figsize=(4.2 * (nk + 2), 4.5))

    axes[0].imshow(img_np)
    axes[0].set_title("Generated Image", weight="bold")
    axes[0].axis("off")

    for i, kw in enumerate(valid_kws):
        ax = axes[i + 1]
        color = PALETTE[i % len(PALETTE)]
        ax.imshow(img_np, alpha=0.35)
        im = ax.imshow(sim_maps[kw], cmap=_single_color_cmap(color), alpha=0.7, vmin=0, vmax=1)
        ax.set_title(f'"{kw}"', weight="bold", color=color)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Combined overlay: each pixel coloured by its dominant keyword
    ax = axes[nk + 1]
    ax.imshow(img_np, alpha=0.45)
    stack = np.stack([sim_maps[k] for k in valid_kws])
    dom_idx = stack.argmax(0)
    dom_val = stack.max(0)
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for i, kw in enumerate(valid_kws):
        c = mcolors.to_rgba(PALETTE[i % len(PALETTE)])
        mask = dom_idx == i
        for ch in range(3):
            overlay[mask, ch] = c[ch]
        overlay[mask, 3] = dom_val[mask] * 0.7
    ax.imshow(overlay)
    ax.set_title("Object Regions", weight="bold")
    ax.axis("off")
    ax.legend(
        handles=[
            mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=kw)
            for i, kw in enumerate(valid_kws)
        ],
        loc="lower right", fontsize=9, framealpha=0.8,
    )

    fig.suptitle("Text → Image Region Correspondence", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Object regions → {save_path}")
    plt.close(fig)


def plot_combined(
    image: Image.Image,
    routing_weights: Optional[torch.Tensor],
    hidden_states: Optional[torch.Tensor],
    tokenizer,
    pos_prompt: str,
    keywords: List[str],
    n_query: int,
    layer_indices: Optional[List[int]] = None,
    save_path: str = "combined.pdf",
):
    """Two-row combined figure: DTR heatmaps (top) + object regions (bottom)."""
    img_np = np.array(image)
    H, W = img_np.shape[:2]
    gs = int(np.sqrt(n_query))

    has_dtr = routing_weights is not None
    has_obj = hidden_states is not None and len(keywords) > 0

    if has_dtr:
        vis_w = routing_weights[0, -n_query:]
        K = vis_w.shape[1]
    else:
        K = 0

    if has_obj:
        sim_maps = _compute_keyword_similarity_maps(
            hidden_states, tokenizer, pos_prompt, keywords, n_query, (H, W)
        )
        valid_kws = list(sim_maps.keys())
    else:
        valid_kws = []

    nrows = int(has_dtr) + int(len(valid_kws) > 0)
    if nrows == 0:
        print("Nothing to visualise in combined figure")
        return

    # ncols = max columns needed across both rows
    dtr_cols = K + 3 if has_dtr else 0     # orig + K layers + dominant + entropy
    obj_cols = len(valid_kws) + 2 if valid_kws else 0  # orig + keywords + combined
    ncols = max(dtr_cols, obj_cols, 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 4.2 * nrows))
    if nrows == 1:
        axes = [axes]
    for row in axes:
        if not hasattr(row, "__len__"):
            row = [row]
        for ax in row:
            ax.axis("off")

    cur_row = 0

    # ── Row: DTR heatmaps ──────────────────────────────
    if has_dtr:
        row_axes = axes[cur_row] if hasattr(axes[cur_row], "__len__") else [axes[cur_row]]
        names = (
            [f"Layer {idx}" for idx in layer_indices]
            if layer_indices
            else [f"Layer {i + 1}" for i in range(K)]
        )

        grid = vis_w.view(gs, gs, K).permute(2, 0, 1).unsqueeze(0).float()
        up = F.interpolate(grid, (H, W), mode="bilinear", align_corners=False).squeeze(0).numpy()

        row_axes[0].imshow(img_np)
        row_axes[0].set_title("Generated Image", weight="bold")

        for k in range(K):
            w = up[k]
            wn = (w - w.min()) / (w.max() - w.min() + 1e-8)
            ax = row_axes[k + 1]
            ax.imshow(img_np, alpha=0.35)
            ax.imshow(wn, cmap="jet", alpha=0.65, vmin=0, vmax=1)
            ax.set_title(names[k])

        # dominant
        dom = vis_w.argmax(-1).view(gs, gs).float()
        dom_up = F.interpolate(dom[None, None], (H, W), mode="nearest").squeeze().numpy()
        ax = row_axes[K + 1]
        ax.imshow(img_np, alpha=0.35)
        cmap_d = plt.cm.get_cmap("Set1", K)
        ax.imshow(dom_up, cmap=cmap_d, alpha=0.6, vmin=-0.5, vmax=K - 0.5)
        ax.set_title("Dominant Layer", weight="bold")
        ax.legend(
            handles=[mpatches.Patch(color=cmap_d(i), label=names[i]) for i in range(K)],
            loc="lower right", fontsize=6, framealpha=0.8,
        )

        # entropy
        eps = 1e-8
        entropy = -(vis_w * (vis_w + eps).log()).sum(-1)
        ent_up = _upsample_grid(entropy, n_query, H, W)
        ent_up = (ent_up - ent_up.min()) / (ent_up.max() - ent_up.min() + 1e-8)
        ax = row_axes[K + 2]
        ax.imshow(img_np, alpha=0.35)
        ax.imshow(ent_up, cmap="magma", alpha=0.65)
        ax.set_title("Entropy", weight="bold")

        # hide unused columns
        for j in range(K + 3, ncols):
            row_axes[j].set_visible(False)

        cur_row += 1

    # ── Row: Object regions ────────────────────────────
    if valid_kws:
        row_axes = axes[cur_row] if hasattr(axes[cur_row], "__len__") else [axes[cur_row]]
        nk = len(valid_kws)

        row_axes[0].imshow(img_np)
        row_axes[0].set_title("Generated Image", weight="bold")

        for i, kw in enumerate(valid_kws):
            ax = row_axes[i + 1]
            color = PALETTE[i % len(PALETTE)]
            ax.imshow(img_np, alpha=0.35)
            ax.imshow(sim_maps[kw], cmap=_single_color_cmap(color), alpha=0.7, vmin=0, vmax=1)
            ax.set_title(f'"{kw}"', weight="bold", color=color)

        # combined overlay
        ax = row_axes[nk + 1]
        ax.imshow(img_np, alpha=0.45)
        stack = np.stack([sim_maps[k] for k in valid_kws])
        dom_idx = stack.argmax(0)
        dom_val = stack.max(0)
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        for i, kw in enumerate(valid_kws):
            c = mcolors.to_rgba(PALETTE[i % len(PALETTE)])
            mask = dom_idx == i
            for ch in range(3):
                overlay[mask, ch] = c[ch]
            overlay[mask, 3] = dom_val[mask] * 0.7
        ax.imshow(overlay)
        ax.set_title("Object Regions", weight="bold")
        ax.legend(
            handles=[
                mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=kw)
                for i, kw in enumerate(valid_kws)
            ],
            loc="lower right", fontsize=7, framealpha=0.8,
        )

        for j in range(nk + 2, ncols):
            row_axes[j].set_visible(False)

    fig.suptitle("Dynamic Token Routing Visualisation", fontsize=16, weight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Combined figure → {save_path}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Per-prompt processing (reused by single & batch modes)
# ────────────────────────────────────────────────────────────
def process_one_prompt(
    model,
    tokenizer,
    pipe: CustomGenPipeline,
    prompt_text: str,
    keywords: List[str],
    output_dir: str,
    guidance_scale: float = 3.1,
    seed: int = 42,
):
    """Generate one image, capture DTR data, and produce all visualisations.

    All outputs are saved under *output_dir* (the caller is responsible for
    creating the directory).
    """
    prompts = build_prompts(prompt_text)
    set_seed(seed)
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)

    with DTRCaptureContext(model) as ctx:
        image = pipe(prompts, guidance_scale=guidance_scale, generator=gen)

    n_query = model.model.config.n_query
    layer_indices = getattr(model.model, "routing_layer_indices", None)

    image.save(os.path.join(output_dir, "generated.png"))

    if ctx.routing_weights is not None:
        plot_dtr_heatmaps(
            image, ctx.routing_weights, n_query,
            layer_indices=layer_indices,
            save_path=os.path.join(output_dir, "dtr_heatmaps.pdf"),
        )

    hs = ctx.hidden_for_similarity
    if keywords and hs is not None:
        plot_object_regions(
            image, hs, tokenizer, prompts[0], keywords, n_query,
            save_path=os.path.join(output_dir, "object_regions.pdf"),
        )

    if ctx.routing_weights is not None or (keywords and hs is not None):
        plot_combined(
            image, ctx.routing_weights, hs,
            tokenizer, prompts[0], keywords, n_query,
            layer_indices=layer_indices,
            save_path=os.path.join(output_dir, "combined.pdf"),
        )


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Visualize Dynamic Token Routing & text-object correspondence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="Path to UniLIP checkpoint")

    # single-prompt mode
    parser.add_argument("--prompt", default=None, help="Text prompt (single-prompt mode)")
    parser.add_argument(
        "--keywords", default="",
        help="Comma-separated keywords (single-prompt mode), e.g. 'cat,sofa,room'",
    )

    # batch mode
    parser.add_argument(
        "--prompt_json", default=None,
        help="Path to a JSON file with a list of {id, prompt, keywords?} dicts (batch mode)",
    )

    parser.add_argument("--guidance_scale", type=float, default=3.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/vis_dtr")
    args = parser.parse_args()

    if args.prompt is None and args.prompt_json is None:
        parser.error("Provide either --prompt (single mode) or --prompt_json (batch mode)")

    # ── Load model (once) ───────────────────────────────
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print("Loading model ...")
    tokenizer, model, _ = load_pretrained_model_general(
        "UniLIP_InternVLForCausalLM", model_path, None, model_name
    )
    model.eval()
    pipe = CustomGenPipeline(multimodal_encoder=model, tokenizer=tokenizer)
    print("Model loaded.")

    # ── Build job list ──────────────────────────────────
    # Each job: (id, prompt_text, keywords_list, per_item_seed)
    jobs: List[tuple] = []

    if args.prompt_json is not None:
        with open(args.prompt_json, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        for item in prompt_data:
            pid = str(item["id"])
            ptxt = item["prompt"]
            kws = [k.strip() for k in item.get("keywords", "").split(",") if k.strip()]
            item_seed = item.get("seed", args.seed)
            jobs.append((pid, ptxt, kws, item_seed))
        print(f"Loaded {len(jobs)} prompts from {args.prompt_json}")
    else:
        kws = [k.strip() for k in args.keywords.split(",") if k.strip()]
        jobs.append(("single", args.prompt, kws, args.seed))

    # ── Process ─────────────────────────────────────────
    base_dir = args.output_dir
    for pid, ptxt, kws, seed in tqdm(jobs, desc="Visualising"):
        out_dir = os.path.join(base_dir, pid) if len(jobs) > 1 else base_dir
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"[{pid}] {ptxt[:80]}{'…' if len(ptxt) > 80 else ''}")
        if kws:
            print(f"  keywords: {kws}")
        print(f"  output  : {out_dir}")

        process_one_prompt(
            model, tokenizer, pipe,
            prompt_text=ptxt,
            keywords=kws,
            output_dir=out_dir,
            guidance_scale=args.guidance_scale,
            seed=seed,
        )

    print(f"\nAll done. Results in {base_dir}")


if __name__ == "__main__":
    main()

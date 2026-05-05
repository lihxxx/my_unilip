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
import textwrap
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
    """Context manager that captures, in a single generation call:
        • DTR router outputs (routing_weights + fused_hidden), if present.
        • LLM backbone last hidden state (fallback similarity source).
        • Per-layer self-attention weights from the bidirectional `llm_connector`
          (the module where visual queries actually attend to text tokens).
        • The `attention_mask` fed into the connector, so we can split
          text vs visual-query positions afterwards.
    """

    def __init__(self, model):
        self.model = model
        self.routing_weights: Optional[torch.Tensor] = None  # [B, N, K]
        self.fused_hidden: Optional[torch.Tensor] = None     # [B, N, D]
        self.last_hidden: Optional[torch.Tensor] = None      # [B, N, D]

        # connector_attentions[layer_idx] : Tensor [B, heads, seq, seq]  (cpu)
        self.connector_attentions: List[torch.Tensor] = []
        # bool mask [B, seq] from the connector's attention_mask input.
        self.connector_token_mask: Optional[torch.Tensor] = None

        self._has_router = (
            getattr(model.model.config, "enable_dynamic_routing", False)
            and hasattr(model.model, "dynamic_router")
        )
        self._orig_router_fwd = None
        self._orig_connector_fwd = None
        self._llm_hook_handle = None
        self._attn_hook_handles: List = []

    # ── enter / exit ──────────────────────────────────────
    def __enter__(self):
        ctx = self

        # Hook 1: replace DynamicTokenRouter.forward
        if self._has_router:
            router = self.model.model.dynamic_router
            self._orig_router_fwd = router.forward

            def _capturing_router_forward(selected_hiddens):
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

            router.forward = _capturing_router_forward

        # Hook 2: main LLM output → last hidden state (similarity fallback).
        def _llm_hook(module, input, output):
            hs = getattr(output, "hidden_states", None)
            if hs is None and isinstance(output, tuple):
                for item in output:
                    if isinstance(item, (tuple, list)) and len(item) > 0 and isinstance(item[0], torch.Tensor):
                        hs = item
                        break
            if hs is not None and len(hs) > 0:
                ctx.last_hidden = hs[-1].detach().cpu()

        self._llm_hook_handle = self.model.model.language_model.register_forward_hook(_llm_hook)

        # Hook 3 + 4: monkey-patch llm_connector.forward to force
        # output_attentions=True, and snoop attention_mask to recover the
        # text/visual-query token split. Then attach a forward hook on every
        # connector layer's self_attn that records its attn_weights.
        connector = getattr(self.model.model, "llm_connector", None)
        if connector is not None:
            self._orig_connector_fwd = connector.forward

            def _capturing_connector_forward(*args, **kwargs):
                # Snoop attention_mask before the call (handles both positional
                # and keyword forms).
                attn_mask = kwargs.get("attention_mask", None)
                if attn_mask is None and len(args) >= 2:
                    attn_mask = args[1]
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        ctx.connector_token_mask = attn_mask.detach().cpu()
                    elif attn_mask.dim() == 2:
                        ctx.connector_token_mask = (attn_mask > 0).detach().cpu()
                    elif attn_mask.dim() == 4:
                        # [B,1,seq,seq] additive mask: a token is valid iff its
                        # diagonal entry is ≥ 0 (i.e. not masked).
                        diag = attn_mask[:, 0].diagonal(dim1=-2, dim2=-1)
                        ctx.connector_token_mask = (diag >= -1.0).detach().cpu()

                # Reset on every call, so we keep only the most recent forward.
                ctx.connector_attentions = []
                kwargs["output_attentions"] = True
                return ctx._orig_connector_fwd(*args, **kwargs)

            connector.forward = _capturing_connector_forward

            def _make_attn_hook(layer_idx: int):
                def _attn_hook(module, inputs, output):
                    # eager-mode HF self_attn returns
                    #   (attn_output, attn_weights[, past_kv]).
                    if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
                        ctx.connector_attentions.append(output[1].detach().to("cpu", torch.float32))
                return _attn_hook

            for li, layer in enumerate(connector.layers):
                attn_module = getattr(layer, "self_attn", None)
                if attn_module is None:
                    continue
                handle = attn_module.register_forward_hook(_make_attn_hook(li))
                self._attn_hook_handles.append(handle)

        return self

    def __exit__(self, *exc):
        if self._has_router and self._orig_router_fwd is not None:
            self.model.model.dynamic_router.forward = self._orig_router_fwd
        if self._llm_hook_handle is not None:
            self._llm_hook_handle.remove()
        if self._orig_connector_fwd is not None:
            self.model.model.llm_connector.forward = self._orig_connector_fwd
        for h in self._attn_hook_handles:
            h.remove()
        self._attn_hook_handles = []

    @property
    def hidden_for_similarity(self) -> Optional[torch.Tensor]:
        """Best available hidden states: routed > last-layer."""
        return self.fused_hidden if self.fused_hidden is not None else self.last_hidden

    def stacked_attentions(self) -> Optional[torch.Tensor]:
        """[L, B, H, S, S] tensor of all connector self-attention layers, or None."""
        if not self.connector_attentions:
            return None
        return torch.stack(self.connector_attentions, dim=0)


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


def _save_fig_dual(fig, save_path: str):
    """Save a matplotlib figure to BOTH .pdf and .png next to each other.

    The server-side renderers cannot preview .pdf, so we always emit a .png
    alongside the .pdf for quick inspection. The .pdf remains the authoritative
    vector copy for the rebuttal.
    """
    base, _ = os.path.splitext(save_path)
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")


def _format_layer_names(K: int, layer_indices: Optional[List[int]] = None) -> List[str]:
    if layer_indices:
        return [f"Layer {int(idx)}" for idx in layer_indices]
    return [f"Layer {i + 1}" for i in range(K)]


def _select_focus_indices(
    vis_w: torch.Tensor,
    layer_indices: Optional[List[int]],
    focus_layers: Optional[List[int]],
    max_layers: int = 2,
) -> List[int]:
    """Choose a small set of layers for compact paper figures.

    If explicit layer ids are supplied, match them against the real routing
    layer ids. Otherwise use the layers with the largest spatial variation;
    those usually produce the most readable routing maps.
    """
    K = vis_w.shape[1]
    if focus_layers:
        layer_values = [int(x) for x in layer_indices] if layer_indices else list(range(1, K + 1))
        selected = []
        for wanted in focus_layers:
            if wanted in layer_values:
                selected.append(layer_values.index(wanted))
            elif 0 <= wanted < K:
                selected.append(wanted)
            elif 1 <= wanted <= K:
                selected.append(wanted - 1)
        selected = list(dict.fromkeys(i for i in selected if 0 <= i < K))
        if selected:
            return selected[:max_layers]

    variation = vis_w.float().std(dim=0)
    return torch.argsort(variation, descending=True)[:max_layers].tolist()


def _relative_map(weight_map: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(weight_map, [2, 98])
    return np.clip((weight_map - lo) / (hi - lo + 1e-8), 0, 1)


def _routing_layer_maps(
    routing_weights: torch.Tensor,
    n_query: int,
    image_hw: tuple,
) -> tuple[torch.Tensor, np.ndarray]:
    vis_w = routing_weights[0, -n_query:].float()  # [N_query, K]
    K = vis_w.shape[1]
    gs = int(np.sqrt(n_query))
    H, W = image_hw
    grid = vis_w.view(gs, gs, K).permute(2, 0, 1).unsqueeze(0)
    up = F.interpolate(grid, (H, W), mode="bilinear", align_corners=False).squeeze(0).numpy()
    return vis_w, up


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


def _gaussian_kernel(sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return np.array([[1.0]])
    radius = max(1, int(round(3.0 * sigma_px)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(coords ** 2) / (2 * sigma_px ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def _smooth(map_2d: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return map_2d
    k = _gaussian_kernel(sigma_px)
    pad = k.shape[0] // 2
    padded = np.pad(map_2d, pad, mode="edge")
    out = np.zeros_like(map_2d)
    H, W = map_2d.shape
    for i in range(H):
        for j in range(W):
            out[i, j] = float(np.sum(padded[i:i + k.shape[0], j:j + k.shape[1]] * k))
    return out


def _compute_keyword_attention_grids(
    hidden_states: torch.Tensor,
    tokenizer,
    pos_prompt: str,
    keywords: List[str],
    n_query: int,
) -> Dict[str, np.ndarray]:
    """Per-keyword cosine-similarity map at the visual-token grid resolution.

    Returns {keyword: ndarray[gs, gs]} (un-normalised, raw cosine values in [-1,1]).
    """
    hs = hidden_states[0]
    n_text = hs.shape[0] - n_query
    text_h = hs[:n_text]
    vis_h = hs[n_text:]
    gs = int(np.sqrt(n_query))

    grids: Dict[str, np.ndarray] = {}
    vis_norm = F.normalize(vis_h.float(), dim=-1)
    for kw in keywords:
        pos = find_keyword_token_positions(tokenizer, pos_prompt, kw)
        if not pos:
            continue
        kw_vec = F.normalize(text_h[pos].float().mean(0, keepdim=True), dim=-1)
        sim = (vis_norm @ kw_vec.T).squeeze(-1)  # [N_query]
        grids[kw] = sim.view(gs, gs).numpy()
    return grids


def _focus_score(grid: np.ndarray, top_frac: float = 0.1) -> float:
    """Mean of top-k% / mean overall. Higher = more spatially focused."""
    flat = grid.flatten()
    k = max(1, int(round(top_frac * flat.size)))
    top_mean = float(np.sort(flat)[-k:].mean())
    overall_mean = float(flat.mean())
    return top_mean - overall_mean


# ────────────────────────────────────────────────────────────
# Attention rollout (Abnar & Zuidema 2020) for VLM-style maps
# ────────────────────────────────────────────────────────────
def _compute_attention_rollout(
    connector_attentions: torch.Tensor,
    sample_idx: int = 0,
    head_reduce: str = "mean",
    add_residual: bool = True,
) -> Optional[torch.Tensor]:
    """Compute the attention rollout matrix for one sample.

    Rollout (Abnar & Zuidema 2020):
        A_l = 0.5 * attn_l + 0.5 * I     # account for residual stream
        rollout = A_L @ A_{L-1} @ ... @ A_1
    Each row is re-normalised to sum to 1.

    Args
    ----
    connector_attentions : Tensor[L, B, H, S, S]
    sample_idx : which batch element (0 = positive prompt).
    head_reduce : 'mean' or 'max' across heads.

    Returns
    -------
    Tensor[S, S] of float32 on CPU, or None if input is empty.
    """
    if connector_attentions is None or connector_attentions.numel() == 0:
        return None
    L, B, H, S, _ = connector_attentions.shape
    a = connector_attentions[:, sample_idx]       # [L, H, S, S]
    if head_reduce == "max":
        a = a.amax(dim=1)                          # [L, S, S]
    else:
        a = a.mean(dim=1)                          # [L, S, S]

    # Residual identity (each token always attends to itself).
    eye = torch.eye(S, dtype=a.dtype)
    rollout = eye.clone()
    for l in range(L):
        layer = a[l]
        if add_residual:
            layer = 0.5 * layer + 0.5 * eye
            # Re-normalise rows after adding identity (probabilities again).
            layer = layer / layer.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        rollout = layer @ rollout
    # Final normalisation per row.
    rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return rollout


def _extract_keyword_rollout_grids(
    rollout: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    tokenizer,
    pos_prompt: str,
    keywords: List[str],
    n_query: int,
    sample_idx: int = 0,
) -> Dict[str, np.ndarray]:
    """Per-keyword visual-query → keyword-text rollout heatmap (gs × gs)."""
    if rollout is None:
        return {}
    S = rollout.shape[0]
    if token_mask is not None:
        valid = token_mask[sample_idx].bool()
        valid_len = int(valid.sum().item())
    else:
        valid_len = S
    n_text = valid_len - n_query
    if n_text <= 0:
        return {}

    visual_to_text = rollout[n_text:n_text + n_query, :n_text]   # [n_query, n_text]
    gs = int(np.sqrt(n_query))

    grids: Dict[str, np.ndarray] = {}
    for kw in keywords:
        pos = find_keyword_token_positions(tokenizer, pos_prompt, kw)
        pos = [p for p in pos if 0 <= p < n_text]
        if not pos:
            continue
        kw_mass = visual_to_text[:, pos].sum(dim=-1)             # [n_query]
        grids[kw] = kw_mass.view(gs, gs).float().numpy()
    return grids


def plot_attention_rollout(
    image: Image.Image,
    connector_attentions: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    tokenizer,
    pos_prompt: str,
    keywords: List[str],
    n_query: int,
    save_path: str,
    smooth_sigma_px: float = 4.0,
    head_reduce: str = "mean",
):
    """Single-checkpoint VLM-style figure: generated image + per-keyword rollout overlays."""
    if connector_attentions is None or not keywords:
        return
    rollout = _compute_attention_rollout(
        connector_attentions, sample_idx=0, head_reduce=head_reduce, add_residual=True
    )
    grids = _extract_keyword_rollout_grids(
        rollout, token_mask, tokenizer, pos_prompt, keywords, n_query, sample_idx=0
    )
    if not grids:
        print(f"No keyword survived tokenisation; skip rollout figure ({pos_prompt[:40]}…)")
        return

    img_np = np.array(image)
    H, W = img_np.shape[:2]
    nk = len(grids)
    fig, axes = plt.subplots(1, nk + 1, figsize=(2.5 * (nk + 1), 2.7))
    if nk == 0:
        axes = [axes]

    axes[0].imshow(img_np)
    axes[0].set_title("Generated", weight="bold", fontsize=10)
    axes[0].axis("off")

    for ax, (kw, grid) in zip(axes[1:], grids.items()):
        attn = _normalise_attention(grid, (H, W), smooth_sigma_px)
        focus = _focus_score(grid)
        ax.imshow(img_np, alpha=0.45)
        ax.imshow(attn, cmap="turbo", alpha=0.65, vmin=0, vmax=1)
        ax.set_title(f'"{kw}"\nfocus={focus:+.3f}', weight="bold", fontsize=10)
        ax.axis("off")

    fig.text(
        0.5, 0.01,
        "Attention rollout (Abnar & Zuidema 2020) over connector self-attention layers, "
        "visual query → keyword text token. Higher 'focus' = sharper localisation.",
        ha="center", fontsize=8, color="#444444",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    _save_fig_dual(fig, save_path)
    print(f"Attention rollout → {save_path} (+ .png)")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Real cross-attention extraction (visual queries → text tokens)
# ────────────────────────────────────────────────────────────
def _extract_keyword_cross_attention_grids(
    connector_attentions: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    tokenizer,
    pos_prompt: str,
    keywords: List[str],
    n_query: int,
    sample_idx: int = 0,
    layer_range: Optional[tuple] = None,
) -> Dict[str, np.ndarray]:
    """Average cross-attention from every visual query token to each keyword's
    text token positions, taken from the bidirectional `llm_connector`.

    Args
    ----
    connector_attentions : Tensor[L, B, H, S, S]
        Stacked self-attention weights from each connector layer.
    token_mask : Tensor[B, S] of bool, or None
        True for valid tokens; visual queries occupy the last `n_query` valid
        positions. If None, we assume no padding.
    sample_idx : int
        Which batch element to use (0 = positive prompt, 1 = unconditional).
    layer_range : (lo, hi) inclusive layer indices to average. None → all layers.

    Returns
    -------
    {keyword: ndarray[gs, gs]}  raw attention probabilities (sum to 1 along key
    axis); larger = stronger attention to that keyword.
    """
    if connector_attentions is None:
        return {}
    L = connector_attentions.shape[0]
    if layer_range is None:
        lo, hi = 0, L - 1
    else:
        lo, hi = layer_range
        lo = max(0, min(L - 1, lo))
        hi = max(0, min(L - 1, hi))
    if lo > hi:
        lo, hi = hi, lo
    # Average across selected layers and heads → [B, S, S]
    attn = connector_attentions[lo : hi + 1].mean(dim=0)  # [B, H, S, S]
    attn = attn.mean(dim=1)                                # [B, S, S]
    a = attn[sample_idx]                                   # [S, S]

    S = a.shape[0]
    if token_mask is not None:
        valid = token_mask[sample_idx].bool()
        valid_len = int(valid.sum().item())
    else:
        valid_len = S

    # Visual queries occupy the LAST n_query valid positions; text precedes.
    n_text = valid_len - n_query
    if n_text <= 0:
        return {}

    # Restrict to valid portion (drop padded rows/cols).
    a = a[:valid_len, :valid_len]                          # [valid_len, valid_len]
    visual_q = a[n_text:n_text + n_query, :n_text]         # [n_query, n_text]
    gs = int(np.sqrt(n_query))

    grids: Dict[str, np.ndarray] = {}
    for kw in keywords:
        pos = find_keyword_token_positions(tokenizer, pos_prompt, kw)
        if not pos:
            continue
        pos = [p for p in pos if 0 <= p < n_text]
        if not pos:
            continue
        # Sum attention probability mass that visual queries put on this kw.
        kw_attn = visual_q[:, pos].sum(dim=-1)             # [n_query]
        grids[kw] = kw_attn.view(gs, gs).numpy()
    return grids


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
    _save_fig_dual(fig, save_path)
    print(f"DTR heatmaps → {save_path} (+ .png)")
    plt.close(fig)


def plot_dtr_focus(
    image: Image.Image,
    routing_weights: torch.Tensor,
    n_query: int,
    layer_indices: Optional[List[int]] = None,
    focus_layers: Optional[List[int]] = None,
    save_path: str = "dtr_focus.pdf",
    title: str = "Compact DTR Routing",
):
    """Compact, rebuttal-friendly DTR visualisation for one sample.

    This intentionally omits text-object correspondence maps because those are
    only a hidden-state similarity proxy and tend to create noisy pseudo-masks.
    """
    img_np = np.array(image)
    H, W = img_np.shape[:2]
    vis_w, up = _routing_layer_maps(routing_weights, n_query, (H, W))
    K = vis_w.shape[1]
    names = _format_layer_names(K, layer_indices)
    selected = _select_focus_indices(vis_w, layer_indices, focus_layers, max_layers=2)

    ncols = 1 + len(selected)
    fig, axes = plt.subplots(1, ncols, figsize=(2.15 * ncols, 2.45))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(img_np)
    axes[0].set_title("Generated", weight="bold", fontsize=10)
    axes[0].axis("off")

    for ax, idx in zip(axes[1:], selected):
        layer_map = _relative_map(up[idx])
        ax.imshow(img_np, alpha=0.36)
        ax.imshow(layer_map, cmap="turbo", alpha=0.72, vmin=0, vmax=1)
        mean_w = vis_w[:, idx].mean().item()
        ax.set_title(f"{names[idx]}\nmean={mean_w:.2f}", weight="bold", fontsize=10)
        ax.axis("off")

    fig.suptitle(title, y=1.02, fontsize=11, weight="bold")
    fig.text(0.5, 0.01, "Heatmaps show relative spatial routing intensity within each selected layer.",
             ha="center", fontsize=8, color="#444444")
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    _save_fig_dual(fig, save_path)
    print(f"Compact DTR focus → {save_path} (+ .png)")
    plt.close(fig)


def plot_rebuttal_summary(
    records: List[Dict],
    save_path: str,
    focus_layers: Optional[List[int]] = None,
    max_rows: int = 3,
):
    """Multi-sample compact DTR figure intended for the rebuttal page."""
    records = records[:max_rows]
    if not records:
        return

    # Use the first record to decide the displayed layer ids, so columns are
    # consistent across all rows.
    first = records[0]
    first_img = np.array(first["image"])
    first_vis_w, _ = _routing_layer_maps(first["routing_weights"], first["n_query"], first_img.shape[:2])
    selected = _select_focus_indices(first_vis_w, first.get("layer_indices"), focus_layers, max_layers=2)
    names = _format_layer_names(first_vis_w.shape[1], first.get("layer_indices"))

    ncols = 1 + len(selected)
    nrows = len(records)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.05 * ncols, 1.95 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    headers = ["Generated"] + [names[i] for i in selected]
    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=9.5, weight="bold", pad=4)

    for row, rec in enumerate(records):
        image = rec["image"]
        img_np = np.array(image)
        H, W = img_np.shape[:2]
        vis_w, up = _routing_layer_maps(rec["routing_weights"], rec["n_query"], (H, W))
        sample_label = rec.get("id", str(row + 1))
        short_prompt = textwrap.shorten(rec.get("prompt", ""), width=24, placeholder="...")

        ax = axes[row, 0]
        ax.imshow(img_np)
        ax.set_ylabel(f"{sample_label}\n{short_prompt}", rotation=0, ha="right", va="center", fontsize=8)
        ax.axis("off")

        for col, idx in enumerate(selected, start=1):
            ax = axes[row, col]
            layer_map = _relative_map(up[idx])
            ax.imshow(img_np, alpha=0.36)
            ax.imshow(layer_map, cmap="turbo", alpha=0.72, vmin=0, vmax=1)
            ax.text(
                0.02, 0.96, f"mean {vis_w[:, idx].mean().item():.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.68, edgecolor="none", pad=1.5),
            )
            ax.axis("off")

    fig.text(0.5, 0.01, "True router outputs. Heatmaps show relative spatial routing intensity; text-object proxy masks are omitted.",
             ha="center", fontsize=8, color="#444444")
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    _save_fig_dual(fig, save_path)
    print(f"Rebuttal summary DTR figure → {save_path} (+ .png)")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Keyword attention overlay (DTR vs no-DTR comparison)
# ────────────────────────────────────────────────────────────
def _normalise_attention(grid: np.ndarray, image_hw: tuple, smooth_sigma_px: float = 8.0) -> np.ndarray:
    """gs×gs grid → H×W normalised heatmap with optional Gaussian smoothing."""
    H, W = image_hw
    g = torch.from_numpy(grid).float()[None, None]
    up = F.interpolate(g, (H, W), mode="bilinear", align_corners=False).squeeze().numpy()
    if smooth_sigma_px > 0:
        up = _smooth(up, smooth_sigma_px)
    lo, hi = np.percentile(up, [5, 99])
    up = np.clip((up - lo) / (hi - lo + 1e-8), 0, 1)
    return up


def plot_keyword_attention(
    image: Image.Image,
    hidden_states: torch.Tensor,
    tokenizer,
    pos_prompt: str,
    keywords: List[str],
    n_query: int,
    save_path: str,
    smooth_sigma_px: float = 8.0,
):
    """Per-keyword attention map overlay for a single image."""
    img_np = np.array(image)
    H, W = img_np.shape[:2]
    grids = _compute_keyword_attention_grids(hidden_states, tokenizer, pos_prompt, keywords, n_query)
    if not grids:
        print(f"No valid keywords for attention overlay (prompt={pos_prompt[:40]}…)")
        return

    nk = len(grids)
    fig, axes = plt.subplots(1, nk + 1, figsize=(2.4 * (nk + 1), 2.6))
    if nk + 1 == 1:
        axes = [axes]

    axes[0].imshow(img_np)
    axes[0].set_title("Generated", weight="bold", fontsize=10)
    axes[0].axis("off")

    for ax, (kw, grid) in zip(axes[1:], grids.items()):
        attn = _normalise_attention(grid, (H, W), smooth_sigma_px)
        focus = _focus_score(grid)
        ax.imshow(img_np, alpha=0.4)
        ax.imshow(attn, cmap="turbo", alpha=0.65, vmin=0, vmax=1)
        ax.set_title(f'"{kw}"\nfocus={focus:+.2f}', weight="bold", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    _save_fig_dual(fig, save_path)
    print(f"Keyword attention → {save_path} (+ .png)")
    plt.close(fig)


def plot_cross_attention_compare(
    records: List[Dict],
    save_path: str,
    max_keywords_per_sample: int = 2,
    smooth_sigma_px: float = 6.0,
    layer_range: Optional[tuple] = None,
    label_dtr: str = "w/ DTR",
    label_base: str = "w/o DTR",
):
    """Side-by-side **real** cross-attention overlay (DTR vs baseline).

    Each record must contain:
        id, prompt, keywords, image_dtr, image_base,
        attn_dtr, mask_dtr, attn_base, mask_base,
        n_query, tokenizer, pos_prompt
    where attn_* is a Tensor[L, B, H, S, S] and mask_* is Tensor[B, S] (bool).
    """
    if not records:
        return

    # Resolve per-row valid keywords using DTR-side attention.
    resolved: List[List[str]] = []
    for rec in records:
        kws = rec.get("keywords", [])
        grids = _extract_keyword_cross_attention_grids(
            rec["attn_dtr"], rec["mask_dtr"], rec["tokenizer"], rec["pos_prompt"],
            kws, rec["n_query"], sample_idx=0, layer_range=layer_range,
        )
        valid = [kw for kw in kws if kw in grids][:max_keywords_per_sample]
        resolved.append(valid)

    max_kw = max((len(v) for v in resolved), default=0)
    if max_kw == 0:
        print("No keyword survived tokenisation; skipping cross-attention compare figure.")
        return

    nrows = len(records)
    ncols = 2 + 2 * max_kw  # gen-DTR | gen-Base | (kw DTR | kw Base) × max_kw
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.95 * ncols, 2.0 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    headers = [f"Generated\n{label_dtr}", f"Generated\n{label_base}"]
    for slot in range(max_kw):
        headers.append(f"kw{slot + 1}\n{label_dtr}")
        headers.append(f"kw{slot + 1}\n{label_base}")
    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=9.5, weight="bold", pad=4)

    for row, (rec, valid_kws) in enumerate(zip(records, resolved)):
        img_d = np.array(rec["image_dtr"])
        img_b = np.array(rec["image_base"])
        Hd, Wd = img_d.shape[:2]
        Hb, Wb = img_b.shape[:2]

        ax = axes[row, 0]
        ax.imshow(img_d)
        ax.set_ylabel(
            f"{rec.get('id', row + 1)}\n{textwrap.shorten(rec.get('prompt',''), width=22, placeholder='...')}",
            rotation=0, ha="right", va="center", fontsize=8,
        )
        ax.axis("off")

        ax = axes[row, 1]
        ax.imshow(img_b)
        ax.axis("off")

        grids_d = _extract_keyword_cross_attention_grids(
            rec["attn_dtr"], rec["mask_dtr"], rec["tokenizer"], rec["pos_prompt"],
            valid_kws, rec["n_query"], sample_idx=0, layer_range=layer_range,
        )
        grids_b = _extract_keyword_cross_attention_grids(
            rec["attn_base"], rec["mask_base"], rec["tokenizer"], rec["pos_prompt"],
            valid_kws, rec["n_query"], sample_idx=0, layer_range=layer_range,
        )

        for slot in range(max_kw):
            col_d = 2 + 2 * slot
            col_b = col_d + 1
            ax_d = axes[row, col_d]
            ax_b = axes[row, col_b]

            if slot >= len(valid_kws):
                ax_d.set_visible(False)
                ax_b.set_visible(False)
                continue
            kw = valid_kws[slot]

            if kw in grids_d:
                attn = _normalise_attention(grids_d[kw], (Hd, Wd), smooth_sigma_px)
                ax_d.imshow(img_d, alpha=0.4)
                ax_d.imshow(attn, cmap="turbo", alpha=0.7, vmin=0, vmax=1)
                ax_d.text(
                    0.02, 0.96, f'"{kw}"  focus {_focus_score(grids_d[kw]):+.3f}',
                    transform=ax_d.transAxes, ha="left", va="top", fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
                )
            else:
                ax_d.imshow(img_d)
            ax_d.axis("off")

            if kw in grids_b:
                attn = _normalise_attention(grids_b[kw], (Hb, Wb), smooth_sigma_px)
                ax_b.imshow(img_b, alpha=0.4)
                ax_b.imshow(attn, cmap="turbo", alpha=0.7, vmin=0, vmax=1)
                ax_b.text(
                    0.02, 0.96, f'"{kw}"  focus {_focus_score(grids_b[kw]):+.3f}',
                    transform=ax_b.transAxes, ha="left", va="top", fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
                )
            else:
                ax_b.imshow(img_b)
            ax_b.axis("off")

    layer_str = "all layers" if layer_range is None else f"layers {layer_range[0]}-{layer_range[1]}"
    fig.text(
        0.5, 0.005,
        f"Real cross-attention from visual queries to keyword text tokens "
        f"(connector self-attn, head-mean over {layer_str}). "
        f"Higher 'focus' (top-10% mean − overall mean) = sharper localisation.",
        ha="center", fontsize=8, color="#444444",
    )
    plt.tight_layout(rect=(0, 0.025, 1, 1))
    _save_fig_dual(fig, save_path)
    print(f"DTR vs baseline cross-attention → {save_path} (+ .png)")
    plt.close(fig)


def plot_routing_only_compare(
    records: List[Dict],
    save_path: str,
    focus_layers: Optional[List[int]] = None,
    max_layers: int = 2,
):
    """Compact figure showing **only** DTR routing maps next to baseline image.

    Use this when the cross-attention story is hard to read; it makes a weaker
    but unambiguous claim ("DTR performs spatially adaptive layer selection").

    Each record needs: id, prompt, image_dtr, image_base, routing_weights, n_query,
    layer_indices.
    """
    if not records:
        return

    # Pick layers from the first record (consistent columns across rows).
    first = records[0]
    img0 = np.array(first["image_dtr"])
    vis_w0, _ = _routing_layer_maps(first["routing_weights"], first["n_query"], img0.shape[:2])
    selected = _select_focus_indices(vis_w0, first.get("layer_indices"), focus_layers, max_layers=max_layers)
    names = _format_layer_names(vis_w0.shape[1], first.get("layer_indices"))

    nrows = len(records)
    ncols = 2 + len(selected)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.95 * ncols, 2.0 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    headers = ["Generated\nw/ DTR", "Generated\nw/o DTR"] + [f"Routing\n{names[i]}" for i in selected]
    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=9.5, weight="bold", pad=4)

    for row, rec in enumerate(records):
        img_d = np.array(rec["image_dtr"])
        img_b = np.array(rec["image_base"])
        Hd, Wd = img_d.shape[:2]

        ax = axes[row, 0]
        ax.imshow(img_d)
        ax.set_ylabel(
            f"{rec.get('id', row + 1)}\n{textwrap.shorten(rec.get('prompt',''), width=22, placeholder='...')}",
            rotation=0, ha="right", va="center", fontsize=8,
        )
        ax.axis("off")

        ax = axes[row, 1]
        ax.imshow(img_b)
        ax.axis("off")

        vis_w, up = _routing_layer_maps(rec["routing_weights"], rec["n_query"], (Hd, Wd))
        for col_off, idx in enumerate(selected):
            ax = axes[row, 2 + col_off]
            layer_map = _relative_map(up[idx])
            ax.imshow(img_d, alpha=0.36)
            ax.imshow(layer_map, cmap="turbo", alpha=0.72, vmin=0, vmax=1)
            ax.text(
                0.02, 0.96, f"mean {vis_w[:, idx].mean().item():.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
            )
            ax.axis("off")

    fig.text(
        0.5, 0.005,
        "DTR routing weights (right) act as a spatially adaptive layer selector. "
        "Baseline (col 2) lacks this mechanism and uses only the last LLM layer.",
        ha="center", fontsize=8, color="#444444",
    )
    plt.tight_layout(rect=(0, 0.025, 1, 1))
    _save_fig_dual(fig, save_path)
    print(f"Routing-only compare → {save_path} (+ .png)")
    plt.close(fig)


def plot_keyword_compare(
    records: List[Dict],
    save_path: str,
    keywords_per_record: Optional[List[List[str]]] = None,
    max_keywords_per_sample: int = 2,
    smooth_sigma_px: float = 8.0,
    label_dtr: str = "w/ DTR",
    label_base: str = "w/o DTR",
):
    """Side-by-side DTR vs baseline keyword attention overlay.

    Each row uses its own keyword set (taken from `keywords_per_record[i]` or
    `records[i]["keywords"]`). The layout is fixed at:

        | Generated (DTR) | Generated (Base) |
        | kw1 DTR | kw1 Base | kw2 DTR | kw2 Base | ...

    Up to `max_keywords_per_sample` keywords are visualised per row; missing
    keywords (or rows with fewer keywords than the global maximum) leave blank
    cells so columns stay aligned.
    """
    if not records:
        return

    if keywords_per_record is None:
        keywords_per_record = [rec.get("keywords", []) for rec in records]

    # Resolve a per-row keyword list, restricted to keywords actually found in
    # the DTR-side prompt tokens (so the layout reflects what we can plot).
    resolved: List[List[str]] = []
    for rec, kws in zip(records, keywords_per_record):
        grids = _compute_keyword_attention_grids(
            rec["hs_dtr"], rec["tokenizer"], rec["pos_prompt"], kws, rec["n_query"]
        )
        valid = [kw for kw in kws if kw in grids][:max_keywords_per_sample]
        resolved.append(valid)

    max_kw = max((len(v) for v in resolved), default=0)
    if max_kw == 0:
        print("No keyword survived tokenisation in any sample; skipping compare figure.")
        return

    nrows = len(records)
    # 2 generated-image cols + 2 cols per keyword slot.
    ncols = 2 + 2 * max_kw
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85 * ncols, 1.95 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    headers = [f"Generated\n{label_dtr}", f"Generated\n{label_base}"]
    for slot in range(max_kw):
        headers.append(f"kw{slot + 1}\n{label_dtr}")
        headers.append(f"kw{slot + 1}\n{label_base}")
    for ax, header in zip(axes[0], headers):
        ax.set_title(header, fontsize=9.5, weight="bold", pad=4)

    for row, (rec, valid_kws) in enumerate(zip(records, resolved)):
        img_dtr_np = np.array(rec["image_dtr"])
        img_base_np = np.array(rec["image_base"])
        H, W = img_dtr_np.shape[:2]
        Hb, Wb = img_base_np.shape[:2]

        # Generated images.
        ax = axes[row, 0]
        ax.imshow(img_dtr_np)
        ax.set_ylabel(
            f"{rec.get('id', row + 1)}\n{textwrap.shorten(rec.get('prompt',''), width=22, placeholder='...')}",
            rotation=0, ha="right", va="center", fontsize=8,
        )
        ax.axis("off")

        ax = axes[row, 1]
        ax.imshow(img_base_np)
        ax.axis("off")

        grids_dtr = _compute_keyword_attention_grids(
            rec["hs_dtr"], rec["tokenizer"], rec["pos_prompt"], valid_kws, rec["n_query"]
        )
        grids_base = _compute_keyword_attention_grids(
            rec["hs_base"], rec["tokenizer"], rec["pos_prompt"], valid_kws, rec["n_query"]
        )

        for slot in range(max_kw):
            col_dtr = 2 + 2 * slot
            col_base = col_dtr + 1

            ax_d = axes[row, col_dtr]
            ax_b = axes[row, col_base]

            if slot < len(valid_kws):
                kw = valid_kws[slot]

                if kw in grids_dtr:
                    attn = _normalise_attention(grids_dtr[kw], (H, W), smooth_sigma_px)
                    ax_d.imshow(img_dtr_np, alpha=0.4)
                    ax_d.imshow(attn, cmap="turbo", alpha=0.7, vmin=0, vmax=1)
                    ax_d.text(
                        0.02, 0.96, f'"{kw}"  focus {_focus_score(grids_dtr[kw]):+.2f}',
                        transform=ax_d.transAxes, ha="left", va="top", fontsize=7,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
                    )
                else:
                    ax_d.imshow(img_dtr_np)

                if kw in grids_base:
                    attn = _normalise_attention(grids_base[kw], (Hb, Wb), smooth_sigma_px)
                    ax_b.imshow(img_base_np, alpha=0.4)
                    ax_b.imshow(attn, cmap="turbo", alpha=0.7, vmin=0, vmax=1)
                    ax_b.text(
                        0.02, 0.96, f'"{kw}"  focus {_focus_score(grids_base[kw]):+.2f}',
                        transform=ax_b.transAxes, ha="left", va="top", fontsize=7,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
                    )
                else:
                    ax_b.imshow(img_base_np)
            else:
                ax_d.set_visible(False)
                ax_b.set_visible(False)

            ax_d.axis("off")
            ax_b.axis("off")

    fig.text(
        0.5, 0.005,
        f"Per-keyword text→visual cosine attention. Higher 'focus' (top-10% mean − overall mean) "
        f"means sharper localisation. Pairs: {label_dtr} vs {label_base} (same prompt & seed).",
        ha="center", fontsize=8, color="#444444",
    )
    plt.tight_layout(rect=(0, 0.025, 1, 1))
    _save_fig_dual(fig, save_path)
    print(f"DTR vs baseline keyword attention → {save_path} (+ .png)")
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
    _save_fig_dual(fig, save_path)
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
    _save_fig_dual(fig, save_path)
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
    focus_layers: Optional[List[int]] = None,
    skip_object_regions: bool = False,
    image_filename: str = "generated.png",
    smooth_sigma_px: float = 8.0,
    keyword_attention_filename: Optional[str] = "keyword_attention.pdf",
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

    image.save(os.path.join(output_dir, image_filename))

    if ctx.routing_weights is not None:
        plot_dtr_heatmaps(
            image, ctx.routing_weights, n_query,
            layer_indices=layer_indices,
            save_path=os.path.join(output_dir, "dtr_heatmaps.pdf"),
        )
        plot_dtr_focus(
            image, ctx.routing_weights, n_query,
            layer_indices=layer_indices,
            focus_layers=focus_layers,
            save_path=os.path.join(output_dir, "dtr_focus.pdf"),
        )

    hs = ctx.hidden_for_similarity
    if keywords and hs is not None and not skip_object_regions:
        plot_object_regions(
            image, hs, tokenizer, prompts[0], keywords, n_query,
            save_path=os.path.join(output_dir, "object_regions.pdf"),
        )

    if not skip_object_regions and (ctx.routing_weights is not None or (keywords and hs is not None)):
        plot_combined(
            image, ctx.routing_weights, hs,
            tokenizer, prompts[0], keywords, n_query,
            layer_indices=layer_indices,
            save_path=os.path.join(output_dir, "combined.pdf"),
        )

    # Standalone single-checkpoint keyword attention overlay (cosine, kept for ref).
    if keywords and hs is not None and keyword_attention_filename:
        plot_keyword_attention(
            image, hs, tokenizer, prompts[0], keywords, n_query,
            save_path=os.path.join(output_dir, keyword_attention_filename),
            smooth_sigma_px=smooth_sigma_px,
        )

    # VLM-style attention rollout (the real money figure).
    stacked_attn = ctx.stacked_attentions()
    if keywords and stacked_attn is not None:
        rollout_filename = f"rollout_{os.path.splitext(image_filename)[0].replace('generated_', '')}.pdf"
        if rollout_filename in ("rollout_.pdf",):
            rollout_filename = "rollout_attention.pdf"
        plot_attention_rollout(
            image, stacked_attn, ctx.connector_token_mask,
            tokenizer, prompts[0], keywords, n_query,
            save_path=os.path.join(output_dir, rollout_filename),
            smooth_sigma_px=smooth_sigma_px,
        )

    return {
        "image": image,
        "routing_weights": ctx.routing_weights,
        "hs": hs,
        "n_query": n_query,
        "layer_indices": layer_indices,
        "pos_prompt": prompts[0],
        "connector_attentions": stacked_attn,
        "connector_token_mask": ctx.connector_token_mask,
    }


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
    parser.add_argument(
        "--max_prompts", type=int, default=None,
        help="Optional cap for quick rebuttal-figure generation.",
    )
    parser.add_argument(
        "--focus_layers", default="",
        help="Comma-separated routing layer ids to show in compact figures, e.g. '16,20'. "
             "If omitted, layers with the largest spatial variation are selected.",
    )
    parser.add_argument(
        "--skip_object_regions", action="store_true",
        help="Skip noisy text-hidden similarity region maps and the old wide combined figure.",
    )
    parser.add_argument(
        "--rebuttal_ids", default="",
        help="Comma-separated prompt ids to include in output_dir/dtr_rebuttal.{pdf,png}.",
    )
    parser.add_argument(
        "--baseline_model_path", default=None,
        help="Optional path to a non-DTR checkpoint with the same architecture. "
             "When provided, the script runs every prompt twice (same seed) and emits "
             "a side-by-side keyword attention comparison figure.",
    )
    parser.add_argument(
        "--smooth_sigma_px", type=float, default=8.0,
        help="Gaussian smoothing sigma (in pixels) applied to keyword attention overlays.",
    )
    args = parser.parse_args()

    if args.prompt is None and args.prompt_json is None:
        parser.error("Provide either --prompt (single mode) or --prompt_json (batch mode)")

    focus_layers = [int(x.strip()) for x in args.focus_layers.split(",") if x.strip()]
    rebuttal_ids = [x.strip() for x in args.rebuttal_ids.split(",") if x.strip()]

    # ── Build job list (independent of model loading) ──
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
            if args.max_prompts is not None and len(jobs) >= args.max_prompts:
                break
        print(f"Loaded {len(jobs)} prompts from {args.prompt_json}")
    else:
        kws = [k.strip() for k in args.keywords.split(",") if k.strip()]
        jobs.append(("single", args.prompt, kws, args.seed))

    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    def _run_pass(model_path: str, tag: str) -> Dict[str, Dict]:
        """Load *model_path*, run all prompts, return {pid: record}.

        *tag* ∈ {"dtr", "base"} controls per-sample output filenames so the two
        passes never overwrite each other.
        """
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

        results: Dict[str, Dict] = {}
        for pid, ptxt, kws, seed in tqdm(jobs, desc=f"Visualising [{tag}]"):
            out_dir = os.path.join(base_dir, pid) if len(jobs) > 1 else base_dir
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n[{tag}|{pid}] {ptxt[:80]}{'…' if len(ptxt) > 80 else ''}")
            if kws:
                print(f"  keywords: {kws}")
            print(f"  output  : {out_dir}")

            rec = process_one_prompt(
                mdl, tok, pipe,
                prompt_text=ptxt,
                keywords=kws,
                output_dir=out_dir,
                guidance_scale=args.guidance_scale,
                seed=seed,
                focus_layers=focus_layers,
                skip_object_regions=args.skip_object_regions,
                image_filename=f"generated_{tag}.png",
                smooth_sigma_px=args.smooth_sigma_px,
                keyword_attention_filename=f"keyword_attention_{tag}.pdf",
            )
            rec.update({
                "id": pid, "prompt": ptxt, "keywords": kws, "tokenizer": tok,
            })
            results[pid] = rec

        # Free GPU memory before loading the next checkpoint.
        del mdl, pipe, tok
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return results

    # ── Pass 1: DTR checkpoint ──────────────────────────
    dtr_results = _run_pass(args.model_path, "dtr")

    # Single-checkpoint summary figure (pure DTR routing).
    summary_records: List[Dict] = []
    for pid, rec in dtr_results.items():
        if rec.get("routing_weights") is None:
            continue
        if rebuttal_ids and pid not in rebuttal_ids:
            continue
        summary_records.append(rec)
    if summary_records:
        plot_rebuttal_summary(
            summary_records,
            save_path=os.path.join(base_dir, "dtr_rebuttal.pdf"),
            focus_layers=focus_layers,
            max_rows=len(summary_records),
        )

    # ── Pass 2 (optional): Baseline checkpoint ──────────
    if args.baseline_model_path:
        base_results = _run_pass(args.baseline_model_path, "base")

        # Build paired records for the side-by-side comparisons.
        compare_records: List[Dict] = []   # for cosine-based plot (kept for ref)
        crossattn_records: List[Dict] = [] # for real cross-attention plot
        routing_records: List[Dict] = []   # for routing-only plot
        for pid, dtr_rec in dtr_results.items():
            if rebuttal_ids and pid not in rebuttal_ids:
                continue
            base_rec = base_results.get(pid)
            if base_rec is None:
                continue
            kws = dtr_rec.get("keywords", [])

            common = {
                "id": pid,
                "prompt": dtr_rec["prompt"],
                "keywords": kws,
                "image_dtr": dtr_rec["image"],
                "image_base": base_rec["image"],
                "n_query": dtr_rec["n_query"],
                "tokenizer": dtr_rec["tokenizer"],
                "pos_prompt": dtr_rec["pos_prompt"],
            }

            # Legacy cosine compare (kept for diagnostic reference).
            if kws and dtr_rec.get("hs") is not None and base_rec.get("hs") is not None:
                compare_records.append({
                    **common,
                    "hs_dtr": dtr_rec["hs"],
                    "hs_base": base_rec["hs"],
                })

            # Primary figure: real cross-attention.
            if (kws
                    and dtr_rec.get("connector_attentions") is not None
                    and base_rec.get("connector_attentions") is not None):
                crossattn_records.append({
                    **common,
                    "attn_dtr": dtr_rec["connector_attentions"],
                    "mask_dtr": dtr_rec["connector_token_mask"],
                    "attn_base": base_rec["connector_attentions"],
                    "mask_base": base_rec["connector_token_mask"],
                })

            # Routing-only fallback figure.
            if dtr_rec.get("routing_weights") is not None:
                routing_records.append({
                    **common,
                    "routing_weights": dtr_rec["routing_weights"],
                    "layer_indices": dtr_rec.get("layer_indices"),
                })

        if crossattn_records:
            # Last few connector layers usually carry the cleanest signal.
            n_layers = crossattn_records[0]["attn_dtr"].shape[0]
            layer_range = (max(0, n_layers - 3), n_layers - 1)
            plot_cross_attention_compare(
                crossattn_records,
                save_path=os.path.join(base_dir, "cross_attention_compare.pdf"),
                smooth_sigma_px=args.smooth_sigma_px,
                layer_range=layer_range,
            )
            plot_cross_attention_compare(
                crossattn_records,
                save_path=os.path.join(base_dir, "cross_attention_compare_alllayers.pdf"),
                smooth_sigma_px=args.smooth_sigma_px,
                layer_range=None,
            )
        else:
            print("No connector-attention records available for cross-attention compare.")

        if routing_records:
            plot_routing_only_compare(
                routing_records,
                save_path=os.path.join(base_dir, "routing_only_compare.pdf"),
                focus_layers=focus_layers,
                max_layers=2,
            )

        if compare_records:
            plot_keyword_compare(
                compare_records,
                save_path=os.path.join(base_dir, "dtr_vs_baseline_cosine.pdf"),
                smooth_sigma_px=args.smooth_sigma_px,
            )

    print(f"\nAll done. Results in {base_dir}")


if __name__ == "__main__":
    main()

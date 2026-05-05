"""DAAM-style image-text cross-attention visualisation for UniLIP / Sana DiT.

For each prompt we generate one image and capture the cross-attention weights
of every ``transformer_blocks[i].attn2`` in the Sana DiT (the layer where the
noise latent attends to the LLM-produced text+visual conditioning sequence).

We then aggregate the attention over a chosen window of denoising steps and
over all DiT layers and attention heads, slice out the columns that belong to
each *keyword* in the prompt, reshape the 256 query positions back to a 16x16
patch grid and overlay the resulting heat-map on the generated image.

Output figure layout (per prompt)::

    [generated image] [kw1 overlay] [kw2 overlay] [kw3 overlay]

The figure is saved as both .pdf and .png next to each other.

Single checkpoint, no DTR/baseline comparison – we just want a clean
DAAM-style image-text alignment map for the rebuttal.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unilip.constants import *  # noqa: F401,F403
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init
from unilip.mm_utils import get_model_name_from_path
from unilip.pipeline_gen import CustomGenPipeline


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompts(prompt_text: str) -> List[str]:
    """Mirror scripts/visualize_dtr.build_prompts.

    Returns ``[positive, negative]``. ``generate_image`` internally swaps the
    batch order so that the DiT receives ``[uncond, cond]``.
    """
    tpl = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n<img>"
    return [
        tpl.format(input=f"Generate an image: {prompt_text}"),
        tpl.format(input="Generate an image."),
    ]


def _save_fig_dual(fig, save_path: str) -> None:
    base, _ = os.path.splitext(save_path)
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")


# ────────────────────────────────────────────────────────────
# Sana DiT cross-attention monkey-patch
# ────────────────────────────────────────────────────────────
@dataclass
class _CrossAttnCapture:
    """Container holding per-step / per-layer cross-attention for ONE prompt.

    ``attn`` is a list-of-list, indexed as ``attn[step_idx][layer_idx]``,
    each element a CPU float32 tensor of shape ``[H, Q, S]`` for the *cond*
    branch only (we drop the uncond half to halve memory).
    """
    attn: List[List[torch.Tensor]] = field(default_factory=list)
    # Buffer that the patched processor appends to during a single DiT call;
    # at the end of each denoising step we move it into ``attn``.
    _step_buf: List[torch.Tensor] = field(default_factory=list)


class _CapturingSanaCrossAttnProcessor:
    """Drop-in replacement for ``SanaAttnProcessor2_0`` that also stores the
    full attention matrix.

    Implementation mirrors the upstream processor exactly except we use an
    explicit softmax(QK^T / sqrt(d)) instead of ``F.scaled_dot_product_attention``
    so we can keep the attention probabilities.
    """

    def __init__(self, capture: _CrossAttnCapture, layer_idx: int):
        self._capture = capture
        self._layer_idx = layer_idx

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ── shape bookkeeping (mirrors SanaAttnProcessor2_0) ──
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # [B, H, Q, D]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # ── explicit attention so we can keep the probabilities ──
        scale = 1.0 / math.sqrt(head_dim)
        # [B, H, Q, S]
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = attn_scores.softmax(dim=-1)

        # Capture cond branch only. The DiT receives latents in order
        # [uncond, cond] (see generate_image), so cond = batch index 1.
        # We always slice [-1] so that the code is robust to bsz==1 (debug).
        with torch.no_grad():
            cond_probs = attn_probs[-1].detach().to(torch.float32).cpu()  # [H, Q, S]
            self._capture._step_buf.append((self._layer_idx, cond_probs))

        # ── continue forward exactly like SanaAttnProcessor2_0 ──
        out = torch.matmul(attn_probs.to(value.dtype), value)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = out.to(query.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        out = out / attn.rescale_output_factor
        return out


class DiTCrossAttnContext:
    """Context manager: monkey-patch all ``attn2.processor`` on the Sana DiT
    transformer blocks, capture cross-attention for every denoising step.

    After ``__exit__`` the capture is available as
    ``ctx.attn[step_idx][layer_idx] -> Tensor[H, Q, S]``  (cond branch only).
    """

    def __init__(self, model):
        self.model = model
        self.dit = model.model.dit
        self.capture = _CrossAttnCapture()
        self._orig_processors: Dict[int, object] = {}
        self._orig_dit_forward = None

    def __enter__(self):
        # 1. Replace each block's attn2.processor with our capturing one.
        for i, block in enumerate(self.dit.transformer_blocks):
            if getattr(block, "attn2", None) is None:
                continue
            self._orig_processors[i] = block.attn2.processor
            block.attn2.set_processor(_CapturingSanaCrossAttnProcessor(self.capture, i))

        # 2. Wrap dit.forward to detect step boundaries: each forward call =
        # one denoising step (latents.repeat(2,...) is forwarded once per step
        # in sample_images).
        self._orig_dit_forward = self.dit.forward
        cap = self.capture

        def _wrapped_forward(*args, **kwargs):
            cap._step_buf = []
            out = self._orig_dit_forward(*args, **kwargs)
            # End-of-step: gather buf into a layer-indexed list.
            if cap._step_buf:
                # sort by layer_idx
                cap._step_buf.sort(key=lambda x: x[0])
                step = [t for _, t in cap._step_buf]
                cap.attn.append(step)
                cap._step_buf = []
            return out

        self.dit.forward = _wrapped_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore processors and forward.
        for i, proc in self._orig_processors.items():
            self.dit.transformer_blocks[i].attn2.set_processor(proc)
        if self._orig_dit_forward is not None:
            self.dit.forward = self._orig_dit_forward
        return False


# ────────────────────────────────────────────────────────────
# Keyword token-position lookup
# ────────────────────────────────────────────────────────────
def find_keyword_token_positions(tokenizer, full_prompt: str, keyword: str) -> List[int]:
    """Return token positions of *keyword* inside ``tokenizer(full_prompt)``.

    We tokenize the full prompt once (matching what ``generate_image`` does),
    then sliding-window match the keyword token-ids.

    Both with-leading-space and without-leading-space variants are tried so we
    catch keywords that appear mid-sentence ("a fluffy orange cat" → ' cat').
    """
    enc = tokenizer(full_prompt, return_tensors="pt", padding=False)
    full_ids: List[int] = enc.input_ids[0].tolist()

    candidates: List[List[int]] = []
    for variant in (" " + keyword, keyword, keyword.lower(), " " + keyword.lower()):
        ids = tokenizer(variant, add_special_tokens=False).input_ids
        if ids and ids not in candidates:
            candidates.append(ids)

    positions: List[int] = []
    for kw_ids in candidates:
        L = len(kw_ids)
        if L == 0:
            continue
        for i in range(len(full_ids) - L + 1):
            if full_ids[i:i + L] == kw_ids:
                positions.extend(range(i, i + L))
        if positions:
            break

    # Dedup, preserve order.
    seen = set()
    out = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ────────────────────────────────────────────────────────────
# Aggregation
# ────────────────────────────────────────────────────────────
def aggregate_cross_attention(
    capture: _CrossAttnCapture,
    step_indices: List[int],
    layer_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """Average cross-attention over the requested steps & layers & heads.

    Returns a tensor of shape ``[Q, S]`` where Q == 256 and S is the encoder
    sequence length.
    """
    if not capture.attn:
        raise RuntimeError("No cross-attention captured. Did the DiT actually run?")

    n_steps = len(capture.attn)
    n_layers = len(capture.attn[0])
    step_indices = [s for s in step_indices if 0 <= s < n_steps]
    if not step_indices:
        step_indices = list(range(n_steps))

    if layer_indices is None:
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]

    accum: Optional[torch.Tensor] = None
    count = 0
    for s in step_indices:
        for l in layer_indices:
            attn = capture.attn[s][l]  # [H, Q, S]
            attn_mean = attn.mean(dim=0)  # [Q, S]
            if accum is None:
                accum = attn_mean
            else:
                accum = accum + attn_mean
            count += 1
    accum = accum / max(count, 1)
    return accum  # [Q, S]


def keyword_grid(
    attn_qs: torch.Tensor,  # [Q, S]
    kw_positions: List[int],
    grid_size: int = 16,
) -> np.ndarray:
    """Slice keyword columns out of ``attn_qs`` and reshape Q -> H x W."""
    if not kw_positions:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    cols = attn_qs[:, kw_positions].mean(dim=-1)  # [Q]
    grid = cols.view(grid_size, grid_size).numpy().astype(np.float32)
    return grid


# ────────────────────────────────────────────────────────────
# Plotting (DAAM-style row)
# ────────────────────────────────────────────────────────────
def _upsample_overlay(grid: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Bilinear upsample a 16x16 attention grid to target image size."""
    t = torch.from_numpy(grid)[None, None].float()
    H, W = target_hw
    up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def _normalize(arr: np.ndarray) -> np.ndarray:
    a = arr - arr.min()
    m = a.max()
    if m > 1e-8:
        a = a / m
    return a


def plot_daam_row(
    image: Image.Image,
    keywords: List[str],
    keyword_grids: Dict[str, np.ndarray],
    save_path: str,
    cmap: str = "jet",
    overlay_alpha: float = 0.55,
    title: Optional[str] = None,
) -> None:
    """Save a single horizontal row figure: ``[image | kw1 | kw2 | ...]``."""
    img_np = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    H, W = img_np.shape[:2]

    n_panels = 1 + len(keywords)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.4))
    if n_panels == 1:
        axes = [axes]

    axes[0].imshow(img_np)
    axes[0].set_title("Generated", fontsize=12)
    axes[0].axis("off")

    for ax, kw in zip(axes[1:], keywords):
        grid = keyword_grids.get(kw)
        if grid is None or grid.sum() < 1e-12:
            ax.imshow(img_np)
            ax.set_title(f"\"{kw}\"\n(no token match)", fontsize=11, color="red")
            ax.axis("off")
            continue
        heat = _upsample_overlay(grid, (H, W))
        heat = _normalize(heat)
        ax.imshow(img_np)
        ax.imshow(heat, cmap=cmap, alpha=overlay_alpha, vmin=0.0, vmax=1.0)
        ax.set_title(f"\"{kw}\"", fontsize=12)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)

    plt.tight_layout()
    _save_fig_dual(fig, save_path)
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Per-prompt processing
# ────────────────────────────────────────────────────────────
def process_one_prompt(
    model,
    tokenizer,
    pipe: CustomGenPipeline,
    prompt_id: str,
    prompt_text: str,
    keywords: List[str],
    output_dir: str,
    guidance_scale: float = 4.5,
    seed: int = 42,
    step_window: Tuple[float, float] = (0.4, 0.6),
    layer_indices: Optional[List[int]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    prompts = build_prompts(prompt_text)
    set_seed(seed)
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)

    with DiTCrossAttnContext(model) as ctx:
        image = pipe(prompts, guidance_scale=guidance_scale, generator=gen)

    # Save raw generated image.
    image.save(os.path.join(output_dir, "generated.png"))

    # Choose denoising steps (default: middle-band [40%, 60%] of total steps).
    n_steps = len(ctx.capture.attn)
    if n_steps == 0:
        print(f"[{prompt_id}] WARNING: no cross-attention captured.")
        return
    s0 = max(0, int(round(step_window[0] * n_steps)))
    s1 = max(s0 + 1, int(round(step_window[1] * n_steps)))
    step_indices = list(range(s0, s1))

    print(f"[{prompt_id}] captured steps={n_steps}, layers={len(ctx.capture.attn[0])}; "
          f"using steps {s0}..{s1 - 1}")

    attn_qs = aggregate_cross_attention(ctx.capture, step_indices, layer_indices)
    # attn_qs shape [Q=256, S]
    Q, S = attn_qs.shape
    grid_side = int(round(math.sqrt(Q)))
    if grid_side * grid_side != Q:
        print(f"[{prompt_id}] WARNING: Q={Q} is not a perfect square; clamping to "
              f"{grid_side}x{grid_side} (drop trailing).")
        attn_qs = attn_qs[: grid_side * grid_side]

    pos_prompt = prompts[0]
    keyword_grids: Dict[str, np.ndarray] = {}
    for kw in keywords:
        positions = find_keyword_token_positions(tokenizer, pos_prompt, kw)
        if not positions:
            print(f"[{prompt_id}]   keyword '{kw}': NOT FOUND in prompt tokens")
            keyword_grids[kw] = np.zeros((grid_side, grid_side), dtype=np.float32)
            continue
        # Filter positions that fall within the captured S range. The encoder
        # sequence is [text_tokens, latent_queries], so token positions of the
        # raw prompt should always lie in [0, S - n_query). We add a safety
        # check anyway.
        positions = [p for p in positions if 0 <= p < S]
        if not positions:
            keyword_grids[kw] = np.zeros((grid_side, grid_side), dtype=np.float32)
            continue
        keyword_grids[kw] = keyword_grid(attn_qs, positions, grid_size=grid_side)
        print(f"[{prompt_id}]   keyword '{kw}': positions={positions}, "
              f"max={keyword_grids[kw].max():.4f}, mean={keyword_grids[kw].mean():.4f}")

    plot_daam_row(
        image=image,
        keywords=keywords,
        keyword_grids=keyword_grids,
        save_path=os.path.join(output_dir, "daam.pdf"),
        title=f"[{prompt_id}] {prompt_text}",
    )

    # Also dump raw grids for later inspection / re-plotting without re-running.
    np.savez(
        os.path.join(output_dir, "daam_grids.npz"),
        **{f"kw_{i}_{kw}": g for i, (kw, g) in enumerate(keyword_grids.items())},
    )


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DAAM-style image-text cross-attention visualisation",
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt_json", required=True)
    parser.add_argument("--output_dir", default="results/vis_daam")
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument(
        "--step_window", default="0.4,0.6",
        help="Fractional window of denoising steps to average over (e.g. '0.4,0.6').",
    )
    parser.add_argument(
        "--layers", default="",
        help="Comma-separated DiT layer indices to average. Empty = all 20 layers.",
    )
    args = parser.parse_args()

    s_lo, s_hi = (float(x) for x in args.step_window.split(","))
    step_window = (s_lo, s_hi)

    layer_indices: Optional[List[int]] = None
    if args.layers.strip():
        layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    # Load prompt jobs
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

    # Load model
    disable_torch_init()
    mp = os.path.expanduser(args.model_path)
    mname = get_model_name_from_path(mp)
    print(f">>> Loading model: {mp}")
    tok, mdl, _ = load_pretrained_model_general(
        "UniLIP_InternVLForCausalLM", mp, None, mname
    )
    mdl.eval()
    pipe = CustomGenPipeline(multimodal_encoder=mdl, tokenizer=tok)
    print(">>> Model loaded.")

    for pid, ptxt, kws, seed in tqdm(jobs, desc="DAAM"):
        out_dir = os.path.join(args.output_dir, pid)
        try:
            process_one_prompt(
                model=mdl,
                tokenizer=tok,
                pipe=pipe,
                prompt_id=pid,
                prompt_text=ptxt,
                keywords=kws,
                output_dir=out_dir,
                guidance_scale=args.guidance_scale,
                seed=seed,
                step_window=step_window,
                layer_indices=layer_indices,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{pid}] ERROR: {e!r}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

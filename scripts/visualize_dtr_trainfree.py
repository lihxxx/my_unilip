#!/usr/bin/env python
"""
visualize_dtr_trainfree.py — Training-free visualisation of Dynamic Token
Routing (DTR) behaviour for SPAR.

Motivation
----------
Reviewer umgx asks whether DTR truly exploits information from different LLM
layers. The fully trained DTR checkpoint is unavailable, so we approximate
DTR's per-token layer-selection behaviour with three parameter-free proxies
that operate on the K candidate layers DTR is configured to read from:

    P_norm[l, i]  = softmax_l ( ||h^l_i||_2 )
    P_novel[l, i] = softmax_l ( 1 - cos(h^l_i, h^{l-1}_i) )
    P_align[l, i] = softmax_l ( cos(h^l_i, h^{L}_i) )    (default proxy)

These proxies have a clear physical meaning:
  * P_norm   — DTR favours layers carrying strong activations.
  * P_novel  — DTR favours layers that introduce new information.
  * P_align  — DTR favours layers whose representation is best aligned with
               the semantic top layer (used in the ablation row of Tab.~A).

Outputs (per prompt, under <output_dir>/<id>/):
  generated.png            — generated image
  layer_usage_hist.pdf     — argmax layer histogram for each proxy
  token_depth_heatmap.pdf  — token-level dominant-layer map overlaid on image
  layer_similarity.pdf     — cross-layer cosine + proxy frequency curves

Aggregate (under <output_dir>/_aggregate/):
  task_layer_usage.pdf     — task-conditioned (T2I vs Edit vs VQA) histograms
  summary.json             — raw per-task usage counts

Usage
-----
    python scripts/visualize_dtr_trainfree.py \
        --model_path results/unilip_intern_vl_1b_sft_alignment_distill05_D6/checkpoint-2385 \
        --t2i_json scripts/dtr_vis_inputs/prompts_t2i.json \
        --output_dir results/vis_dtr_trainfree
"""

import os
import json
import random
import argparse
from collections import defaultdict
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

from unilip.constants import *
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init
from unilip.mm_utils import get_model_name_from_path
from unilip.pipeline_gen import CustomGenPipeline

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
    "savefig.facecolor": "white",
})

PROXY_NAMES = ["norm", "novel", "align"]
PROXY_TITLES = {
    "norm":  "Activation magnitude proxy",
    "novel": "Cross-layer novelty proxy",
    "align": "Top-layer alignment proxy",
}


# ────────────────────────────────────────────────────────────
# Hidden-state capture (hooks LLM, no trained DTR required)
# ────────────────────────────────────────────────────────────
class HiddenStateCapture:
    """Capture the full tuple of LLM hidden states from the last forward call."""

    def __init__(self, model):
        self.model = model
        self.hidden_states: Optional[tuple] = None
        self._handle = None

    def __enter__(self):
        ctx = self

        def _hook(module, inputs, output):
            hs = getattr(output, "hidden_states", None)
            if hs is None and isinstance(output, tuple):
                for item in output:
                    if isinstance(item, (tuple, list)) and len(item) > 0 \
                            and isinstance(item[0], torch.Tensor):
                        hs = item
                        break
            if hs is not None:
                ctx.hidden_states = tuple(t.detach().float().cpu() for t in hs)

        self._handle = self.model.model.language_model.register_forward_hook(_hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()


# ────────────────────────────────────────────────────────────
# Training-free DTR proxies
# ────────────────────────────────────────────────────────────
def select_candidate_layers(num_layers: int, K: int) -> List[int]:
    """Replicate DynamicTokenRouter's candidate-layer schedule."""
    step = max(1, num_layers // K)
    idx = [step * (i + 1) for i in range(K)]
    idx[-1] = num_layers
    return idx


def compute_proxy_weights(layer_hiddens: List[torch.Tensor],
                          last_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return {proxy_name: [N, K]} softmax weights for a single sample.

    layer_hiddens : list length K of [N, D]
    last_hidden   : [N, D]   (final LLM hidden state)
    """
    K = len(layer_hiddens)
    stacked = torch.stack(layer_hiddens, dim=1)        # [N, K, D]
    last_n = F.normalize(last_hidden.float(), dim=-1)  # [N, D]

    # 1) magnitude
    mag = stacked.float().norm(dim=-1)                 # [N, K]
    # 2) novelty against previous candidate layer
    prev = torch.cat([stacked[:, :1], stacked[:, :-1]], dim=1)
    novel = 1.0 - F.cosine_similarity(stacked.float(), prev.float(), dim=-1)
    # 3) alignment with final hidden state
    stacked_n = F.normalize(stacked.float(), dim=-1)
    align = (stacked_n * last_n.unsqueeze(1)).sum(-1)  # [N, K]

    return {
        "norm":  F.softmax(mag,   dim=-1),
        "novel": F.softmax(novel, dim=-1),
        "align": F.softmax(align, dim=-1),
    }


# ────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────
def plot_layer_usage_hist(weights: Dict[str, torch.Tensor],
                          layer_indices: List[int],
                          save_path: str):
    """Argmax layer histogram for each proxy on visual tokens."""
    K = len(layer_indices)
    fig, axes = plt.subplots(1, len(PROXY_NAMES), figsize=(4.0 * len(PROXY_NAMES), 3.6))
    x = np.arange(K)
    labels = [f"L{idx}" for idx in layer_indices]
    for ax, name in zip(axes, PROXY_NAMES):
        w = weights[name]                       # [N_query, K]
        argmax = w.argmax(-1).numpy()
        counts = np.bincount(argmax, minlength=K) / max(len(argmax), 1)
        ax.bar(x, counts, color="#3b6cb7", edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Candidate LLM layer")
        ax.set_ylabel("Token fraction (argmax)")
        ax.set_title(PROXY_TITLES[name])
        for i, c in enumerate(counts):
            ax.text(i, c + 0.02, f"{c:.2f}", ha="center", fontsize=9)
    fig.suptitle("Per-token argmax layer distribution (training-free DTR proxy)",
                 weight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  layer_usage_hist → {save_path}")


def plot_token_depth_heatmap(image: Image.Image,
                             weights: Dict[str, torch.Tensor],
                             layer_indices: List[int],
                             save_path: str):
    """Per-proxy token-level dominant-layer map overlaid on the generated image."""
    img_np = np.array(image)
    H, W = img_np.shape[:2]
    K = len(layer_indices)

    fig, axes = plt.subplots(1, len(PROXY_NAMES) + 1,
                             figsize=(4.0 * (len(PROXY_NAMES) + 1), 4.4))
    axes[0].imshow(img_np)
    axes[0].set_title("Generated Image", weight="bold")
    axes[0].axis("off")

    cmap = plt.cm.get_cmap("Set1", K)
    for ax, name in zip(axes[1:], PROXY_NAMES):
        w = weights[name]                       # [N, K]
        N = w.shape[0]
        gs = int(np.sqrt(N))
        if gs * gs != N:
            ax.text(0.5, 0.5, f"N={N} not square", ha="center", va="center")
            ax.axis("off")
            continue
        dom = w.argmax(-1).view(gs, gs).float()
        dom_up = F.interpolate(dom[None, None], (H, W), mode="nearest").squeeze().numpy()
        ax.imshow(img_np, alpha=0.35)
        ax.imshow(dom_up, cmap=cmap, alpha=0.65, vmin=-0.5, vmax=K - 0.5)
        ax.set_title(PROXY_TITLES[name])
        ax.axis("off")
        handles = [mpatches.Patch(color=cmap(i), label=f"L{layer_indices[i]}")
                   for i in range(K)]
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)

    fig.suptitle("Token-level dominant layer (training-free DTR proxy)",
                 weight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  token_depth_heatmap → {save_path}")


def plot_layer_similarity(layer_hiddens_full: List[torch.Tensor],
                          weights: Dict[str, torch.Tensor],
                          layer_indices: List[int],
                          save_path: str):
    """Cross-layer cosine similarity curve over ALL layers + proxy frequencies
    on candidate layers."""
    L = len(layer_hiddens_full) - 1   # exclude embedding row
    sims = []
    for l in range(1, L + 1):
        a = F.normalize(layer_hiddens_full[l].float(), dim=-1)
        b = F.normalize(layer_hiddens_full[l - 1].float(), dim=-1)
        sims.append((a * b).sum(-1).mean().item())

    fig, ax1 = plt.subplots(figsize=(7.0, 3.8))
    xs = np.arange(1, L + 1)
    ax1.plot(xs, sims, color="#444", linewidth=1.6, label="cos(h^l, h^{l-1})")
    ax1.set_xlabel("LLM layer index l")
    ax1.set_ylabel("Adjacent-layer cosine similarity", color="#444")
    ax1.set_ylim(min(sims) - 0.02, 1.005)
    ax1.set_title("Layer information change vs. proxy layer preference",
                  weight="bold")

    ax2 = ax1.twinx()
    K = len(layer_indices)
    bar_w = 0.25
    colors = {"norm": "#e6194b", "novel": "#3cb44b", "align": "#4363d8"}
    for j, name in enumerate(PROXY_NAMES):
        w = weights[name]
        freq = np.bincount(w.argmax(-1).numpy(), minlength=K) / max(w.shape[0], 1)
        offset = (j - 1) * bar_w
        ax2.bar(np.array(layer_indices) + offset, freq, width=bar_w,
                color=colors[name], alpha=0.7, label=name)
    ax2.set_ylabel("Token fraction selecting layer (proxy)", color="#444")
    ax2.set_ylim(0, 1.0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  layer_similarity → {save_path}")


def plot_task_usage(task_to_counts: Dict[str, Dict[str, np.ndarray]],
                    layer_indices: List[int],
                    save_path: str):
    """Task-conditioned histograms for the default proxy ('align')."""
    proxy = "align"
    tasks = list(task_to_counts.keys())
    K = len(layer_indices)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6),
                             squeeze=False)
    axes = axes[0]
    x = np.arange(K)
    labels = [f"L{idx}" for idx in layer_indices]
    for ax, task in zip(axes, tasks):
        counts = task_to_counts[task][proxy]
        counts = counts / max(counts.sum(), 1)
        ax.bar(x, counts, color="#7a3aa8", edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Candidate LLM layer")
        ax.set_ylabel("Token fraction (argmax)")
        ax.set_title(task)
        for i, c in enumerate(counts):
            ax.text(i, c + 0.02, f"{c:.2f}", ha="center", fontsize=9)
    fig.suptitle(f"Task-conditioned layer preference ({PROXY_TITLES[proxy]})",
                 weight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"task_layer_usage → {save_path}")


# ────────────────────────────────────────────────────────────
# Per-prompt processing
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


def process_one(model, tokenizer, pipe, prompt_text, output_dir,
                K=4, guidance_scale=3.1, seed=42):
    set_seed(seed)
    gen = torch.Generator(device=model.device).manual_seed(seed)
    prompts = build_prompts(prompt_text)

    with HiddenStateCapture(model) as cap:
        image = pipe(prompts, guidance_scale=guidance_scale, generator=gen)
    if cap.hidden_states is None:
        print("  ! no hidden states captured, skip")
        return None

    n_query = model.model.config.n_query
    num_layers = model.model.config.text_config.num_hidden_layers
    layer_indices = select_candidate_layers(num_layers, K)

    # take positive prompt (idx 0) only, last n_query tokens are the visual queries
    layer_h = [cap.hidden_states[idx][0, -n_query:] for idx in layer_indices]
    last_h = cap.hidden_states[-1][0, -n_query:]
    weights = compute_proxy_weights(layer_h, last_h)

    image.save(os.path.join(output_dir, "generated.png"))

    plot_layer_usage_hist(weights, layer_indices,
                          os.path.join(output_dir, "layer_usage_hist.pdf"))
    plot_token_depth_heatmap(image, weights, layer_indices,
                             os.path.join(output_dir, "token_depth_heatmap.pdf"))
    plot_layer_similarity(
        [cap.hidden_states[i][0, -n_query:] for i in range(len(cap.hidden_states))],
        weights, layer_indices,
        os.path.join(output_dir, "layer_similarity.pdf"))

    # return per-proxy argmax counts for aggregation
    counts = {}
    K_ = len(layer_indices)
    for name in PROXY_NAMES:
        counts[name] = np.bincount(weights[name].argmax(-1).numpy(),
                                   minlength=K_).astype(np.int64)
    return {"layer_indices": layer_indices, "counts": counts}


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def load_jobs(t2i_json: Optional[str]):
    jobs = []
    if t2i_json and os.path.exists(t2i_json):
        with open(t2i_json) as f:
            for item in json.load(f):
                jobs.append(("t2i", str(item["id"]), item["prompt"]))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", default="UniLIP_InternVLForCausalLM")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--t2i_json", default="scripts/dtr_vis_inputs/prompts_t2i.json")
    parser.add_argument("--output_dir", default="results/vis_dtr_trainfree")
    parser.add_argument("--K", type=int, default=4,
                        help="number of DTR candidate layers (matches paper)")
    parser.add_argument("--guidance_scale", type=float, default=3.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    print("Loading model ...")
    tokenizer, model, _ = load_pretrained_model_general(args.cls, model_path)
    model.eval()
    pipe = CustomGenPipeline(multimodal_encoder=model, tokenizer=tokenizer)

    jobs = load_jobs(args.t2i_json)
    print(f"Loaded {len(jobs)} T2I jobs.")

    os.makedirs(args.output_dir, exist_ok=True)

    K = args.K
    task_counts = defaultdict(lambda: defaultdict(lambda: np.zeros(K, dtype=np.int64)))
    layer_indices_used = None

    for task, pid, ptxt in tqdm(jobs, desc="Visualising"):
        out_dir = os.path.join(args.output_dir, f"{task}_{pid}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{task}/{pid}] {ptxt[:80]}{'…' if len(ptxt) > 80 else ''}")
        try:
            res = process_one(model, tokenizer, pipe, ptxt, out_dir,
                              K=K, guidance_scale=args.guidance_scale,
                              seed=args.seed)
        except Exception as e:
            print(f"  ! failed: {e}")
            continue
        if res is None:
            continue
        layer_indices_used = res["layer_indices"]
        for name, c in res["counts"].items():
            task_counts[task][name] += c

    # aggregate
    agg_dir = os.path.join(args.output_dir, "_aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    if layer_indices_used is not None and len(task_counts) > 0:
        plot_task_usage({t: dict(d) for t, d in task_counts.items()},
                        layer_indices_used,
                        os.path.join(agg_dir, "task_layer_usage.pdf"))
        summary = {
            "layer_indices": layer_indices_used,
            "task_counts": {t: {n: c.tolist() for n, c in d.items()}
                            for t, d in task_counts.items()},
        }
        with open(os.path.join(agg_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"summary → {os.path.join(agg_dir, 'summary.json')}")

    print(f"\nAll done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()

#!/bin/bash
# UniLIP Dual Stream Training (ImageNet): Training with Dual Stream Architecture
# 
# Architecture:
# Stream 1 (Semantic): vit_embeds -> TransformerEncoder -> semantic_feat (for distill_loss)
#                      -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
# Stream 2 (Pixel): vit_embeds -> down_blocks -> down_mlp -> 32-dim
# Fusion: concat(32, 32) -> 64-dim -> fusion_layer -> 32-dim -> latent_for_decoder
#
# Key differences from PS-VAE:
# - semantic_feat is used for distillation loss (instead of vit_embeds)
# - Two parallel streams for better disentanglement of semantic and pixel features
# - Fusion layer to combine both streams

source /mnt/tidal-alsh01/dataset/zeus/lihongxiang/network.sh

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WORKSPACE="./"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

config_file=./configs/training/InternVL3_1B_DCAE/internvl3_1B_dual_stream_IN.yaml

echo "Running UniLIP Dual Stream Training (ImageNet) on Node $RANK / $WORLD_SIZE (Master: $MASTER_ADDR:$MASTER_PORT)"

torchrun \
    --nproc_per_node=8 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    config=$config_file

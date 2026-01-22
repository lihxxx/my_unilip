#!/bin/bash
# UniLIP PS-VAE Stage 2 (ImageNet): Training with Semantic AE
# 目标: 224px ImageNet 图像重建 + 特征蒸馏 + PS-VAE语义重建
# 
# Based on paper: https://arxiv.org/pdf/2512.17909
# PS-VAE maps representation features to compact latent space with:
# - Semantic Encoder: 3 Transformer blocks + MLP projection
# - Semantic Decoder: 3 Transformer blocks + MLP projection  
# - Semantic reconstruction loss (L2 + cosine similarity)
# - KL divergence regularization

source /mnt/tidal-alsh01/dataset/zeus/lihongxiang/network.sh

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WORKSPACE="./"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

config_file=./configs/training/InternVL3_1B_DCAE/internvl3_1B_psvae_stage2_imagenet.yaml

echo "Running UniLIP PS-VAE Stage 2 (ImageNet) on Node $RANK / $WORLD_SIZE (Master: $MASTER_ADDR:$MASTER_PORT)"

torchrun \
    --nproc_per_node=8 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    config=$config_file

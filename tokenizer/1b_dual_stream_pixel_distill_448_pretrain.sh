#!/bin/bash
# UniLIP Dual Stream Training with Pixel Distillation (ImageNet)
# 
# Architecture:
# Stream 1 (Semantic): vit_embeds -> TransformerEncoder -> semantic_feat (for distill_loss)
#                      -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
# Stream 2 (Pixel): vit_embeds -> down_blocks -> down_mlp -> 32-dim (for pixel_distill_loss)
# Fusion: concat(32, 32) -> 64-dim -> fusion_layer -> 32-dim -> latent_for_decoder
#
# Loss Functions:
# 1. Semantic Distill Loss: MSE(semantic_feat, frozen_vit_feat)
#    - semantic_feat is the output of semantic TransformerEncoder
#    - frozen_vit_feat is from frozen InternVL3-1B ViT
#
# 2. Pixel Distill Loss: MSE(pixel_latent, frozen_dcae_latent)
#    - pixel_latent (32-dim) is from pixel stream
#    - frozen_dcae_latent (32-dim) is from frozen DC-AE encoder
#
# 3. Reconstruction Loss: MSE/L1 between input and reconstructed images
#
# 4. Perceptual Loss: LPIPS loss using ConvNeXt-S

source /mnt/tidal-alsh01/dataset/zeus/lihongxiang/network.sh

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WORKSPACE="./"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

config_file=./configs/training/InternVL3_1B_DCAE/internvl3_1B_dual_stream_pixel_distill_448_IN2.yaml

echo "Running UniLIP Dual Stream with Pixel Distillation (ImageNet) on Node $RANK / $WORLD_SIZE (Master: $MASTER_ADDR:$MASTER_PORT)"

torchrun \
    --nproc_per_node=8 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    config=$config_file

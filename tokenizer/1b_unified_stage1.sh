#!/bin/bash
# UniLIP Stage 1: Train decoder with frozen encoder
# 目标: 图像重建 (冻结 ViT encoder)

source /mnt/tidal-alsh01/dataset/zeus/lihongxiang/network.sh

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WORKSPACE="./"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

config_file=./configs/training/InternVL3_1B_DCAE/internvl3_1B_unified_stage1.yaml

echo "Running UniLIP Stage 1 on Node $RANK / $WORLD_SIZE (Master: $MASTER_ADDR:$MASTER_PORT)"

torchrun \
    --nproc_per_node=8 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    config=$config_file

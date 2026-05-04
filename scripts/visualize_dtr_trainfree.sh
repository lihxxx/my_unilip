#!/bin/bash

MODEL="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/results/unilip_intern_vl_1b_sft_alignment_distill05_D6/checkpoint-2385"
CLS_NAME="UniLIP_InternVLForCausalLM"

CUDA_VISIBLE_DEVICES=0 python scripts/visualize_dtr_trainfree.py \
    --cls "$CLS_NAME" \
    --model_path "$MODEL" \
    --t2i_json scripts/dtr_vis_inputs/prompts_t2i.json \
    --output_dir results/vis_dtr_trainfree \
    --K 4 \
    --guidance_scale 3.1 \
    --seed 42

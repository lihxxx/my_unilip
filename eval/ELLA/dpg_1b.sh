#!/bin/bash

MODEL="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/results/unilip_intern_vl_1b_sft/checkpoint-2385"
CLS_NAME="UniLIP_InternVLForCausalLM"

# 请修改为你的 DPG-Bench prompts 真实所在路径
PROMPT_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/eval_bench/ELLA/dpg_bench/prompts"

# Total number of GPUs/chunks.
N_CHUNKS=2

# Launch processes in parallel for each GPU/chunk.
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching DPG-Bench process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python /mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/eval/ELLA/dpg.py \
        --cls "$CLS_NAME" \
        --model "$MODEL" \
        --prompt_dir "$PROMPT_DIR" \
        --index $i \
        --n_chunks $N_CHUNKS &
done

# Wait for all background processes to finish.
wait
echo "All DPG-Bench background processes finished."
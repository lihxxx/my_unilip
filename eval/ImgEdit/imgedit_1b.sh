#!/bin/bash





MODEL="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/results/unilip_intern_vl_1b_sft_alignment_distill05_D6/checkpoint-2385"
CLS_NAME="UniLIP_InternVLForCausalLM"

# Total number of GPUs/chunks.
N_CHUNKS=2

# Launch processes in parallel for each GPU/chunk.
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python imgedit.py --cls "$CLS_NAME" --model "$MODEL" --index $i --n_chunks $N_CHUNKS &
done

# Wait for all background processes to finish.
wait
echo "All background processes finished."



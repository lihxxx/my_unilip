#!/bin/bash


MODEL="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/models/UniLIP-3B"
CLS_NAME="UniLIP_InternVLForCausalLM"
METHOD="spar_2b"
OUTDIR="results"


# Total number of GPUs/chunks.
N_CHUNKS=4

# Launch processes in parallel for each GPU/chunk.
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python gedit.py --cls "$CLS_NAME" --model "$MODEL" \
        --method_name "$METHOD" --outdir "$OUTDIR" --scale 4.0 \
        --index $i --n_chunks $N_CHUNKS &
done

# Wait for all background processes to finish.
wait
echo "All background processes finished."

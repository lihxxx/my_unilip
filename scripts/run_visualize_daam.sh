#!/bin/bash
#
# DAAM-style image-text cross-attention visualisation
# (Sana DiT cross-attn, single checkpoint, no DTR/baseline split).
#
# For each prompt we save:
#   ${OUTPUT_DIR}/${pid}/generated.png
#   ${OUTPUT_DIR}/${pid}/daam.{pdf,png}      <- the row figure for the rebuttal
#   ${OUTPUT_DIR}/${pid}/daam_grids.npz      <- raw 16x16 keyword grids

set -e

export BASE_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip"
MODEL_PATH="${BASE_DIR}/results/unilip_intern_vl_1b_sft_alignment_distill05_D6_dynamic6/checkpoint-2385"
PROMPT_JSON="${BASE_DIR}/prompts_vis.json"
OUTPUT_DIR="${BASE_DIR}/results/vis_daam"

GUIDANCE_SCALE=4.5
SEED=42
MAX_PROMPTS=7
# Average over the middle 20% of denoising steps (DAAM heuristic).
STEP_WINDOW="0.4,0.6"
# Empty = average all 20 DiT layers.
LAYERS=""

cd "${BASE_DIR}"

echo "============================================"
echo "  DAAM Cross-Attention Visualisation"
echo "  Model     : ${MODEL_PATH}"
echo "  Prompts   : ${PROMPT_JSON} (max ${MAX_PROMPTS})"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Guidance  : ${GUIDANCE_SCALE}"
echo "  Steps win : ${STEP_WINDOW}"
echo "  Layers    : ${LAYERS:-all}"
echo "============================================"

python scripts/visualize_daam.py \
    --model_path "${MODEL_PATH}" \
    --prompt_json "${PROMPT_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    --guidance_scale ${GUIDANCE_SCALE} \
    --seed ${SEED} \
    --max_prompts ${MAX_PROMPTS} \
    --step_window "${STEP_WINDOW}" \
    --layers "${LAYERS}"

echo "Done. Results saved under ${OUTPUT_DIR}/<id>/daam.{pdf,png}"

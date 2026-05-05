#!/bin/bash
#
# DAAM-style cross-attention comparison: Baseline (UniLIP-1B) vs DTR (dynamic6).
# Same seed, same prompts, two rows per figure.
#
# For each prompt we save:
#   ${OUTPUT_DIR}/${pid}/generated_base.png
#   ${OUTPUT_DIR}/${pid}/generated_dtr.png
#   ${OUTPUT_DIR}/${pid}/daam_grids_base.npz
#   ${OUTPUT_DIR}/${pid}/daam_grids_dtr.npz
#   ${OUTPUT_DIR}/${pid}/daam_compare.{pdf,png}   <- the rebuttal figure

set -e

export BASE_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip"
BASELINE_MODEL_PATH="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/models/UniLIP-1B"
DTR_MODEL_PATH="${BASE_DIR}/results/unilip_intern_vl_1b_sft_alignment_distill05_D6_dynamic6/checkpoint-2385"
PROMPT_JSON="${BASE_DIR}/prompts_vis.json"
OUTPUT_DIR="${BASE_DIR}/results/vis_daam_compare"

GUIDANCE_SCALE=4.5
SEED=42
MAX_PROMPTS=7
STEP_WINDOW="0.4,0.6"
LAYERS=""

cd "${BASE_DIR}"

echo "============================================"
echo "  DAAM Compare (Baseline vs DTR)"
echo "  Baseline  : ${BASELINE_MODEL_PATH}"
echo "  DTR       : ${DTR_MODEL_PATH}"
echo "  Prompts   : ${PROMPT_JSON} (max ${MAX_PROMPTS})"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Guidance  : ${GUIDANCE_SCALE}"
echo "  Seed      : ${SEED}"
echo "  Steps win : ${STEP_WINDOW}"
echo "  Layers    : ${LAYERS:-all}"
echo "============================================"

python scripts/visualize_daam_compare.py \
    --baseline_model_path "${BASELINE_MODEL_PATH}" \
    --dtr_model_path "${DTR_MODEL_PATH}" \
    --prompt_json "${PROMPT_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    --guidance_scale ${GUIDANCE_SCALE} \
    --seed ${SEED} \
    --max_prompts ${MAX_PROMPTS} \
    --step_window "${STEP_WINDOW}" \
    --layers "${LAYERS}" \
    --baseline_label "Baseline (UniLIP-1B)" \
    --dtr_label "DTR (Ours)"

echo "Done. Comparison figures saved under ${OUTPUT_DIR}/<id>/daam_compare.{pdf,png}"

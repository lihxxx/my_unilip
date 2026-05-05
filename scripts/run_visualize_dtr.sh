#!/bin/bash
#
# DTR 路由可视化脚本
# 生成真实 Dynamic Token Routing 权重，并输出 rebuttal-friendly compact figure。
#

# ============== 路径配置 ==============
export BASE_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip"
MODEL_PATH="${BASE_DIR}/results/unilip_intern_vl_1b_sft_alignment_distill05_D6_dynamic6/checkpoint-2385"
PROMPT_JSON="${BASE_DIR}/prompts_vis.json"
OUTPUT_DIR="${BASE_DIR}/results/vis_dtr"

# ============== 生成参数 ==============
GUIDANCE_SCALE=3.1
SEED=42
MAX_PROMPTS=7

# 当前 dynamic6 checkpoint 的候选层为 4,8,12,16,20,24。
# Layer 16/20 在现有样例中空间结构最清晰，适合 1 页 rebuttal 展示。
FOCUS_LAYERS="16,20"
REBUTTAL_IDS="001,003,007"

cd "${BASE_DIR}"

echo "============================================"
echo "  DTR Visualisation"
echo "  Model : ${MODEL_PATH}"
echo "  Prompts: ${PROMPT_JSON}"
echo "  Output : ${OUTPUT_DIR}"
echo "  Focus  : Layer ${FOCUS_LAYERS}"
echo "  Fig IDs: ${REBUTTAL_IDS}"
echo "============================================"

python scripts/visualize_dtr.py \
    --model_path "${MODEL_PATH}" \
    --prompt_json "${PROMPT_JSON}" \
    --guidance_scale ${GUIDANCE_SCALE} \
    --seed ${SEED} \
    --output_dir "${OUTPUT_DIR}" \
    --max_prompts ${MAX_PROMPTS} \
    --focus_layers "${FOCUS_LAYERS}" \
    --rebuttal_ids "${REBUTTAL_IDS}" \
    --skip_object_regions

echo "Done. Results saved to ${OUTPUT_DIR}"
echo "Compact rebuttal figure: ${OUTPUT_DIR}/dtr_rebuttal.pdf and .png"

#!/bin/bash
#
# DTR 路由可视化脚本
# 批量生成图像并可视化 Dynamic Token Routing 权重 + 文本-物体对应
#

# ============== 路径配置 ==============
export BASE_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip"
MODEL_PATH="${BASE_DIR}/results/unilip_intern_vl_1b_sft_alignment_distill05_D6_dynamic6/checkpoint-2385"
PROMPT_JSON="${BASE_DIR}/prompts_vis.json"
OUTPUT_DIR="${BASE_DIR}/results/vis_dtr"

# ============== 生成参数 ==============
GUIDANCE_SCALE=3.1
SEED=42

cd "${BASE_DIR}"

echo "============================================"
echo "  DTR Visualisation"
echo "  Model : ${MODEL_PATH}"
echo "  Prompts: ${PROMPT_JSON}"
echo "  Output : ${OUTPUT_DIR}"
echo "============================================"

python scripts/visualize_dtr.py \
    --model_path "${MODEL_PATH}" \
    --prompt_json "${PROMPT_JSON}" \
    --guidance_scale ${GUIDANCE_SCALE} \
    --seed ${SEED} \
    --output_dir "${OUTPUT_DIR}"

echo "Done. Results saved to ${OUTPUT_DIR}"

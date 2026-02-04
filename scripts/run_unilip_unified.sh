#!/bin/bash
#
# 统一训练脚本
# 支持Stage 1/2/3训练和Alignment Distill损失
#
# Usage:
#   bash run_unilip_unified.sh --stage 1              # Stage1训练
#   bash run_unilip_unified.sh --stage 2              # Stage2训练
#   bash run_unilip_unified.sh --stage 3              # Stage3 SFT
#   bash run_unilip_unified.sh --stage 3 --repa       # Stage3 + Alignment Distill
#   bash run_unilip_unified.sh --stage 3 --repa-e     # Stage3 + Alignment Distill (解冻vision encoder)
#

set -e

# ============== 默认配置 ==============
STAGE=2
ENABLE_REPA=False
UNFREEZE_VISION_ENCODER=False
REPA_LOSS_WEIGHT=0.5
REPA_ENCODER_DEPTH=6

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --repa)
            ENABLE_REPA=True
            shift
            ;;
        --repa-e)
            ENABLE_REPA=True
            UNFREEZE_VISION_ENCODER=True
            shift
            ;;
        --repa-weight)
            REPA_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --repa-depth)
            REPA_ENCODER_DEPTH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Training Configuration:"
echo "  Stage: ${STAGE}"
echo "  Enable Alignment Distill: ${ENABLE_REPA}"
echo "  Unfreeze Vision Encoder: ${UNFREEZE_VISION_ENCODER}"
if [ "$ENABLE_REPA" = "True" ]; then
    echo "  Alignment Distill Loss Weight: ${REPA_LOSS_WEIGHT}"
    echo "  Alignment Encoder Depth: ${REPA_ENCODER_DEPTH}"
fi
echo "============================================"

# ============== 环境变量 ==============
# 请根据实际情况修改以下路径
export BASE_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip"
export MODEL_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/models"

# ============== 模型路径配置 ==============
if [ "$STAGE" = "1" ]; then
    # Stage 1: 从HuggingFace加载基础模型
    export MODEL_NAME_OR_PATH="OpenGVLab/InternVL3-1B-hf"
    export WANDB_NAME="unilip_1b_stage1"
    export FIX_DIT=True
    export FIX_CONNECT=False
    export GEN_IMG_FOLDER="${BASE_DIR}/data/BLIP3o-Pretrain"
    export EDIT_IMG_FOLDER=""
    export GEN_REPEAT=1
    export EDIT_REPEAT=0
    export IMAGE_SIZE=512
elif [ "$STAGE" = "2" ]; then
    # Stage 2: 从Stage 1 checkpoint加载
    export MODEL_NAME_OR_PATH="${BASE_DIR}/work_dirs/1b_stage1/checkpoint-xxx"  # 请修改为实际checkpoint路径
    export WANDB_NAME="unilip_1b_stage2"
    export FIX_DIT=False
    export FIX_CONNECT=False
    export GEN_IMG_FOLDER="${BASE_DIR}/data/BLIP3o-Pretrain"
    export EDIT_IMG_FOLDER="${BASE_DIR}/data/GPT-Edit"
    export GEN_REPEAT=1
    export EDIT_REPEAT=10
    export IMAGE_SIZE=448
else
    # Stage 3 (SFT): 从Stage 2 checkpoint加载
    export MODEL_NAME_OR_PATH="${MODEL_DIR}/UniLIP-1B"  # 请修改为实际checkpoint路径
    export WANDB_NAME="unilip_1b_stage3_sft"
    export FIX_DIT=False
    export FIX_CONNECT=False
    export GEN_IMG_FOLDER="${BASE_DIR}/data/gen_sft"
    export EDIT_IMG_FOLDER="${BASE_DIR}/data/edit_sft"
    export GEN_REPEAT=1
    export EDIT_REPEAT=3
    export IMAGE_SIZE=448
fi

# Alignment Distill模式下更新名称
if [ "$ENABLE_REPA" = "True" ]; then
    if [ "$UNFREEZE_VISION_ENCODER" = "True" ]; then
        export WANDB_NAME="${WANDB_NAME}_align_distill_e"
    else
        export WANDB_NAME="${WANDB_NAME}_align_distill"
    fi
fi

export OUTPUT_FOLDER="${BASE_DIR}/results/${WANDB_NAME}"

# ============== WandB配置 ==============
unset WANDB_DISABLED
export WANDB_API_KEY="your_wandb_api_key_here"  # 请填入你的WandB API Key
export WANDB_PROJECT="unilip_umm"

# ============== 构建训练命令 ==============
TRAIN_CMD="torchrun --nproc_per_node=8"

# 多节点训练支持
if [ -n "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    TRAIN_CMD="${TRAIN_CMD} --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_port=${MASTER_PORT} --master_addr=${MASTER_ADDR}"
else
    TRAIN_CMD="${TRAIN_CMD} --master_port=29506"
fi

TRAIN_CMD="${TRAIN_CMD} unilip/train/train_unified.py"

# ============== 基础参数 ==============
TRAIN_CMD="${TRAIN_CMD} \
    --deepspeed ${BASE_DIR}/deepspeed_scripts/zero0.json \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --unilip_path ${MODEL_DIR}/UniLIP/1b_unilip.pth \
    --unilip_factor 10.6 \
    --mllm_path ${MODEL_DIR}/InternVL3-1B \
    --mllm_hf_path ${MODEL_DIR}/InternVL3-1B-hf \
    --vae_path ${MODEL_DIR}/dc-ae-f32c32-sana-1.1-diffusers \
    --dit_path ${MODEL_DIR}/Sana_600M_512px_diffusers \
    --version internvl \
    --data_type mix \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_FOLDER} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{\"min_lr\":1e-5}' \
    --model_max_length 1024 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --n_query 256 \
    --n_und_query 0 \
    --report_to wandb \
    --run_name ${WANDB_NAME} \
    --stage ${STAGE} \
    --fix_dit ${FIX_DIT} \
    --fix_connect ${FIX_CONNECT} \
    --fix_llm True \
    --image_size ${IMAGE_SIZE}"

# ============== 数据路径参数 ==============
if [ -n "$GEN_IMG_FOLDER" ]; then
    TRAIN_CMD="${TRAIN_CMD} --gen_image_folder ${GEN_IMG_FOLDER}"
fi

if [ -n "$EDIT_IMG_FOLDER" ] && [ "$EDIT_IMG_FOLDER" != "" ]; then
    TRAIN_CMD="${TRAIN_CMD} --edit_image_folder ${EDIT_IMG_FOLDER}"
fi

TRAIN_CMD="${TRAIN_CMD} --gen_repeat ${GEN_REPEAT}"

if [ "$EDIT_REPEAT" -gt 0 ]; then
    TRAIN_CMD="${TRAIN_CMD} --edit_repeat ${EDIT_REPEAT}"
fi

# ============== Alignment Distill参数 ==============
if [ "$ENABLE_REPA" = "True" ]; then
    TRAIN_CMD="${TRAIN_CMD} \
        --enable_repa True \
        --repa_loss_weight ${REPA_LOSS_WEIGHT} \
        --repa_encoder_depth ${REPA_ENCODER_DEPTH}"
    
    if [ "$UNFREEZE_VISION_ENCODER" = "True" ]; then
        TRAIN_CMD="${TRAIN_CMD} --unfreeze_vision_encoder True"
    fi
fi

# ============== 执行训练 ==============
echo ""
echo "Running command:"
echo "${TRAIN_CMD}"
echo ""

eval ${TRAIN_CMD}

#!/bin/bash
#
# UniLIP 1B 统一训练脚本示例
#
# 使用说明:
#   1. 修改下方路径配置
#   2. 选择训练模式（Stage 1/2/3，是否启用REPA）
#   3. 运行: bash run_unilip_1b_unified.sh
#

# ============== 路径配置（请根据实际情况修改）==============
BASE_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip"
MODEL_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/models"

# WandB配置
export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_PROJECT="unilip_umm"

# ============== 训练模式选择 ==============
# 取消注释你想要运行的模式

# Stage 1: 初始训练，只训练connector，冻结dit
# bash run_unilip_unified.sh --stage 1

# Stage 2: 预训练，训练connector + dit
# bash run_unilip_unified.sh --stage 2

# Stage 3: SFT微调
# bash run_unilip_unified.sh --stage 3

# Stage 3 + REPA: SFT微调 + REPA损失（类似原版REPA，冻结vision encoder）
# bash run_unilip_unified.sh --stage 3 --repa --repa-weight 0.5 --repa-depth 6

# Stage 3 + REPA-E: SFT微调 + REPA损失（类似REPA-E，解冻vision encoder）
# bash run_unilip_unified.sh --stage 3 --repa-e --repa-weight 0.5 --repa-depth 6

# ============== 默认运行Stage 3 + REPA ==============
unset WANDB_DISABLED
export WANDB_NAME="unilip_intern_vl_1b_sft_repa"
export OUTPUT_FOLDER="${BASE_DIR}/results/${WANDB_NAME}"
export GEN_IMG_FOLDER="${BASE_DIR}/data/gen_sft"
export EDIT_IMG_FOLDER="${BASE_DIR}/data/edit_sft"

# 单节点训练
# torchrun --nproc_per_node=8 --master_port=29506 \

# 多节点训练
torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR \
    unilip/train/train_unified.py \
    --deepspeed ${BASE_DIR}/deepspeed_scripts/zero0.json \
    --model_name_or_path ${MODEL_DIR}/UniLIP-1B \
    --unilip_path ${MODEL_DIR}/UniLIP/1b_unilip.pth \
    --unilip_factor 10.6 \
    --mllm_path ${MODEL_DIR}/InternVL3-1B \
    --mllm_hf_path ${MODEL_DIR}/InternVL3-1B-hf \
    --vae_path ${MODEL_DIR}/dc-ae-f32c32-sana-1.1-diffusers \
    --dit_path ${MODEL_DIR}/Sana_600M_512px_diffusers \
    --version internvl \
    --data_type "mix" \
    --gen_image_folder ${GEN_IMG_FOLDER} \
    --edit_image_folder ${EDIT_IMG_FOLDER} \
    --gen_repeat 1 \
    --edit_repeat 3 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_FOLDER} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
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
    --stage 3 \
    --fix_dit False \
    --fix_connect False \
    --fix_llm True \
    --enable_repa True \
    --repa_loss_weight 0.5 \
    --repa_encoder_depth 6 \
    --unfreeze_vision_encoder False

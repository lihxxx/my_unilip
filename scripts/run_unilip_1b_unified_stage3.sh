#!/bin/bash
#
# UniLIP 1B Stage3 统一训练脚本
# Stage3: SFT微调阶段，可选启用REPA损失
#

# ============== 路径配置（请根据实际情况修改）==============
export BASE_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip"
export MODEL_DIR="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/models"

# 数据路径支持多个目录，用逗号分隔
export GEN_IMG_FOLDER="${BASE_DIR}/data/gen_sft"
export EDIT_IMG_FOLDER="${BASE_DIR}/data/edit_sft"

# ============== WandB配置 ==============
unset WANDB_DISABLED
export WANDB_API_KEY="3ed65eb52edcc37a5e278a82dd874b44d4ffadb7"
export WANDB_PROJECT="unilip_umm"
export WANDB_NAME="unilip_intern_vl_1b_sft_self_repa"
export OUTPUT_FOLDER="${BASE_DIR}/results/${WANDB_NAME}"

# ============== REPA配置 ==============
ENABLE_REPA=True
REPA_LOSS_WEIGHT=0.5
REPA_ENCODER_DEPTH=6
UNFREEZE_VISION_ENCODER=False

# ============== 训练命令 ==============
# 单节点训练
# torchrun --nproc_per_node=8 --master_port=29506 \

# 多节点训练
torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR \
    unilip/train/train_unified.py \
    --deepspeed ${BASE_DIR}/deepspeed_scripts/zero0.json \
    --model_name_or_path ${MODEL_DIR}/UniLIP-1B \
    --use_vae_model False \
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
    --image_size 448 \
    --use_vae_image_norm False \
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
    --fix_dit False \
    --fix_connect False \
    --fix_llm True \
    --enable_repa ${ENABLE_REPA} \
    --repa_loss_weight ${REPA_LOSS_WEIGHT} \
    --repa_encoder_depth ${REPA_ENCODER_DEPTH} \
    --unfreeze_vision_encoder ${UNFREEZE_VISION_ENCODER}

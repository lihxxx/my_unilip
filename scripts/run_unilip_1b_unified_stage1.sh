#!/bin/bash
#
# UniLIP 1B Stage1 统一训练脚本
# Stage1: 训练connector，冻结DiT，只使用生成数据
#

# ============== 路径配置（请根据实际情况修改）==============
export BASE_DIR="../"
export OUTPUT_FOLDER="${BASE_DIR}/work_dirs/1b_unified_stage1"

# 数据路径：多个目录用逗号分隔
export GEN_IMG_FOLDER="${BASE_DIR}/data/BLIP3o-Pretrain-Long-Caption,${BASE_DIR}/data/BLIP3o-Pretrain-Short-Caption,${BASE_DIR}/data/BLIP3o-Pretrain-JourneyDB"

# ============== WandB配置 ==============
unset WANDB_DISABLED
export WANDB_API_KEY="3ed65eb52edcc37a5e278a82dd874b44d4ffadb7"
export WANDB_PROJECT="unilip_umm"
export WANDB_NAME="unilip_intern_vl_1b_sft_self_repa_w05xd6_lastlayer"
export OUTPUT_FOLDER="${BASE_DIR}/results/${WANDB_NAME}"

# ============== 训练命令 ==============
# 单节点训练
# torchrun --nproc_per_node=8 --master_port=29506 \

# 多节点训练
torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR \
    unilip/train/train_unified.py \
    --deepspeed ${BASE_DIR}/deepspeed_scripts/zero0.json \
    --model_name_or_path OpenGVLab/InternVL3-1B-hf \
    --use_vae_model True \
    --unilip_path ${BASE_DIR}/tokenizer_ckpt/1b_unilip.pth \
    --unilip_factor 10.6 \
    --mllm_path ${MODEL_DIR}/InternVL3-1B \
    --mllm_hf_path ${MODEL_DIR}/InternVL3-1B-hf \
    --vae_path ${MODEL_DIR}/dc-ae-f32c32-sana-1.1-diffusers \
    --dit_path ${MODEL_DIR}/Sana_600M_512px_diffusers \
    --version internvl \
    --data_type "mix" \
    --gen_image_folder ${GEN_IMG_FOLDER} \
    --gen_repeat 1 \
    --image_size 512 \
    --use_vae_image_norm True \
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
    --report_to none \
    --run_name unilip_1b_stage1 \
    --fix_dit True \
    --fix_connect False \
    --fix_llm True

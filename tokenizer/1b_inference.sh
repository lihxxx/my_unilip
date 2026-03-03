WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9997 --same_network \
    scripts/inference.py config=/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/tokenizer/checkpoints/64GCF_bs4stage2F64stage2_1b_dualstreamPixTrans616_10x00_448E256_pretrain_GAN00k_lr3e4x01E1e5W10k_100k/config.yaml \
    checkpoint_path=/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/tokenizer/checkpoints/64GCF_bs4stage2F64stage2_1b_dualstreamPixTrans616_10x00_448E256_pretrain_GAN00k_lr3e4x01E1e5W10k_100k/checkpoint-100000/ema_model/pytorch_model.bin \
    model.use_padding=False \
    img_path="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/demo/edit_input.jpg" \
    dataset.preprocessing.resize_shorter_edge=448 dataset.preprocessing.crop_size=448

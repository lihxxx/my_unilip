WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=2 --machine_rank=0 --main_process_ip=127.0.0.2 --main_process_port=9999 --same_network \
    scripts/evaluation.py config=/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/tokenizer/checkpoints/unilip_stage2_224_imagenet_from_scratch/config.yaml \
    checkpoint_path=/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/tokenizer/checkpoints/unilip_stage2_224_imagenet_from_scratch/checkpoint-90000/ema_model/pytorch_model.bin \
    model.use_padding=True \
    training.per_gpu_batch_size=32 experiment.eval_max_samples=-1 experiment.eval_every=1 \
    dataset.preprocessing.resize_shorter_edge=256 dataset.preprocessing.crop_size=256 \
    lr_scheduler.warmup_steps=100000 lr_scheduler.end_lr=1e-10 optimizer.learning_rate=1e-10 optimizer.discriminator_learning_rate=1e-10 optimizer.weight_decay=1e-10
    


    # experiment.project="1B_stage2" experiment.name="1B_stage2" experiment.output_dir="1B_stage2" training.per_gpu_batch_size=16
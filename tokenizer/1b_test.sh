WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=2 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9997 --same_network \
    scripts/evaluation.py config=/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/tokenizer/configs/training/InternVL3_1B_DCAE/internvl3_1B_psvae_IN.yaml \
    checkpoint_path="/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/tokenizer/checkpoints/unilip_1b_psvae_224_IN_midC96/checkpoint-30000/ema_model/pytorch_model.bin" \
    training.per_gpu_batch_size=256 experiment.eval_max_samples=-1 experiment.eval_every=1 \
    dataset.preprocessing.resize_shorter_edge=256 dataset.preprocessing.crop_size=256 \


    # experiment.project="1B_stage2" experiment.name="1B_stage2" experiment.output_dir="1B_stage2" training.per_gpu_batch_size=16
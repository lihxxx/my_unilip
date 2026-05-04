# cd eval/GEdit-Bench && bash gedit_1b.sh

cd /mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/Step1X-Edit/GEdit-Bench
python run_gedit_score.py --model_name spar_1b \
    --edited_images_dir /mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/eval/GEdit-Bench/results \
    --save_dir score_dir --backbone qwen25vl
python calculate_statistics.py --model_name spar_1b \
    --save_path score_dir --backbone qwen25vl --language all
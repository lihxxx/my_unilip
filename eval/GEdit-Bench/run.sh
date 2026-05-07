# cd eval/GEdit-Bench && bash gedit_1b.sh
# NOTE: scoring backbone is GPT-4.1 via OpenAI API.
# Make sure Step1X-Edit/GEdit-Bench/secret.env contains your OPENAI key (first line).

cd /mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/Step1X-Edit/GEdit-Bench
python run_gedit_score.py --model_name spar_1b \
    --edited_images_dir /mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/eval/GEdit-Bench/results \
    --save_dir score_dir --backbone gpt4o
python calculate_statistics.py --model_name spar_1b \
    --save_path score_dir --backbone gpt4o --language all
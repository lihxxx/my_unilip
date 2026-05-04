import argparse
import os
import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor

from unilip.constants import *
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init
from unilip.pipeline_edit import CustomEditPipeline


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_template(prompt):
    instruction = ('<|im_start|>user\n{input}<|im_end|>\n'
                   '<|im_start|>assistant\n<img>')
    pos_prompt = instruction.format(input=prompt[0])
    cfg_prompt = instruction.format(input=prompt[1])
    return [pos_prompt, cfg_prompt]


torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, default="", help="CLASS NAME")
    parser.add_argument("--model", type=str, required=True, help="checkpoint path")
    parser.add_argument("--prompt_template", type=str, default="qwen", help="Template format")
    parser.add_argument("--method_name", type=str, default="spar",
                        help="method name used as the result subdir, must match --model_name in run_gedit_score.py")
    parser.add_argument("--outdir", type=str, default="results",
                        help="root output dir; final layout: {outdir}/{method_name}/fullset/{task}/{cn,en}/{key}.png")
    parser.add_argument("--language", type=str, default="all", choices=["all", "en", "cn"])
    parser.add_argument("--steps", type=int, default=50, help="number of sampling steps")
    parser.add_argument("--scale", type=float, default=4.5, help="cfg scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="stepfun-ai/GEdit-Bench")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--index", type=int, default=0, help="Chunk index to process (0-indexed)")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total number of chunks")
    return parser.parse_args()


def main(opt):
    disable_torch_init()
    tokenizer, multi_model, _ = load_pretrained_model_general(opt.cls, opt.model)
    image_processor = AutoProcessor.from_pretrained(multi_model.config.mllm_hf_path).image_processor
    pipe = CustomEditPipeline(multimodal_encoder=multi_model, tokenizer=tokenizer, image_processor=image_processor)

    # Load GEdit-Bench
    dataset = load_dataset(opt.dataset_name)[opt.dataset_split]

    # Filter by language and shard across chunks
    items = []
    for item in dataset:
        if opt.language != "all" and item['instruction_language'] != opt.language:
            continue
        items.append(item)
    items = items[opt.index::opt.n_chunks]
    print(f"Processing chunk {opt.index}/{opt.n_chunks}, {len(items)} samples assigned.")

    method_root = os.path.join(opt.outdir, opt.method_name, "fullset")
    os.makedirs(method_root, exist_ok=True)
    generator = torch.Generator(device=multi_model.device).manual_seed(opt.seed)

    for idx, item in enumerate(tqdm(items)):
        set_global_seed(seed=opt.seed)
        key = item['key']
        task_type = item['task_type']
        language = item['instruction_language']
        instruction = item['instruction']

        out_dir = os.path.join(method_root, task_type, language)
        os.makedirs(out_dir, exist_ok=True)
        outpath = os.path.join(out_dir, f"{key}.png")
        if os.path.exists(outpath):
            continue

        prompt = [f"Edit the image: {instruction}\n<image>", "Edit the image.\n<image>"]
        if "qwen" in opt.prompt_template:
            multimodal_prompts = add_template(prompt)
        else:
            multimodal_prompts = prompt
        print(f"[{idx:>4}/{len(items)}] task={task_type} lang={language} key={key}")

        input_image = item['input_image_raw'].convert("RGB")
        multimodal_prompts.append(input_image)

        with torch.no_grad():
            gen_img = pipe(multimodal_prompts, guidance_scale=opt.scale, generator=generator)
            gen_img.save(outpath)

    print("Done.")


if __name__ == "__main__":
    main(parse_args())

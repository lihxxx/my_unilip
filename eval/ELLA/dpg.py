import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_tensor, to_pil_image
import random

from unilip.constants import *
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init
from unilip.pipeline_gen import CustomGenPipeline

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
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Huggingface model name")
    parser.add_argument("--prompt_template", type=str, default="qwen", help="Template format")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Directory containing DPG-Bench txt prompts")
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to")
    parser.add_argument("--scale", type=float, default=4.5, help="unconditional guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--index", type=int, default=0, help="Chunk index to process (0-indexed)")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total number of chunks")
    opt = parser.parse_args()
    return opt

def main(opt):
    model_name = opt.model
    # 将输出目录命名为 dpg_bench，方便区分
    outdir = f"{model_name}/dpg_bench_{opt.prompt_template}"
    os.makedirs(outdir, exist_ok=True)
    
    disable_torch_init()
    tokenizer, multi_model, context_len = load_pretrained_model_general(opt.cls, model_name)
    pipe = CustomGenPipeline(multimodal_encoder=multi_model, tokenizer=tokenizer)

    # 读取目录下所有的 txt 文件
    prompt_files = sorted(glob.glob(os.path.join(opt.prompt_dir, "*.txt")))
    
    # 按照 n_chunks 切分数据，分配给当前进程
    prompt_files = prompt_files[opt.index::opt.n_chunks]
    print(f"Processing chunk {opt.index} out of {opt.n_chunks} total chunks, {len(prompt_files)} prompts assigned.")

    for index, file_path in enumerate(prompt_files):
        # 读取 TXT 文件中的 prompt 文本
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
            
        # 获取文件名 (不含后缀) 用于保存对应的图像
        base_filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_filename)[0]
        outpath = os.path.join(outdir, f"{name_without_ext}.png")

        # 构建模型输入
        prompt_formatted = [f"Generate an image: {prompt_text}", "Generate an image."]
        if "qwen" in opt.prompt_template:
            prompt_formatted = add_template(prompt_formatted)
            
        print(f"Prompt ({index: >3}/{len(prompt_files)}) File: {base_filename}")
        
        img_tensors = []
        # 为当前 prompt 生成 4 张图像
        for i in range(4):
            # 给定不同的 seed 以保证 4 张图的多样性
            current_seed = opt.seed + index * 4 + i
            set_global_seed(current_seed)
            
            with torch.no_grad():
                gen_img = pipe(prompt_formatted, guidance_scale=opt.scale)
                # 转为 Tensor 以便后续利用 torchvision.utils.make_grid 拼接
                img_tensors.append(to_tensor(gen_img))
        
        # 将 4 张图拼接为 2x2 的网格格式 (nrow=2)
        grid_tensor = make_grid(img_tensors, nrow=2, padding=0)
        grid_pil = to_pil_image(grid_tensor)
        
        # 按照要求以原始 txt 的文件名保存网格图片
        grid_pil.save(outpath)

    print(f"Chunk {opt.index} Done.")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
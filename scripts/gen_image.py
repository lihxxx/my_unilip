from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
import torch
import sys
import os
import json
from tqdm import tqdm
from unilip.constants import *
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init
from unilip.mm_utils import get_model_name_from_path
from unilip.pipeline_gen import CustomGenPipeline
import random

# 初始化模型路径
model_path = "/mnt/tidal-alsh01/dataset/zeus/lihongxiang/unified_model/my_unilip/results/unilip_intern_vl_1b_sft_alignment_distill05_D6"
disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)

# 加载模型
print("Loading model...")
tokenizer, multi_model, context_len = load_pretrained_model_general('UniLIP_InternVLForCausalLM', model_path, None, model_name)
pipe = CustomGenPipeline(multimodal_encoder=multi_model, tokenizer=tokenizer)
print("Model loaded successfully.")

def create_image_grid(images, rows, cols):
    """Creates a grid of images and returns a single PIL Image."""
    assert len(images) == rows * cols
    width, height = images[0].size
    grid_width = width * cols
    grid_height = height * rows
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for i, image in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        grid_image.paste(image, (x, y))

    return grid_image

def add_template(prompt):
    instruction = ('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n<img>')
    pos_prompt = instruction.format(input=prompt[0])
    cfg_prompt = instruction.format(input=prompt[1])
    return [pos_prompt, cfg_prompt]

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- 修改部分开始 ---

# 1. 创建 results 文件夹
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# 2. 读取 JSON 文件
json_file_path = "prompts.json" # 请确保这个文件和脚本在同一目录下
with open(json_file_path, "r", encoding="utf-8") as f:
    prompt_data = json.load(f)

# 设置全局种子
set_global_seed(seed=1024)
base_generator = torch.Generator(device=multi_model.device)

# 3. 遍历 Prompt 数据生成图片
for item in tqdm(prompt_data, desc="Processing Prompts"):
    prompt_id = item["id"]
    prompt_text = item["prompt"]
    
    gen_images = []
    print(f"\nGenerating for ID {prompt_id}: {prompt_text[:50]}...")
    
    # 每个 prompt 生成 4 张图
    for i in range(4):
        # 为了保证这4张图不一样，基于基础种子加入循环变量 i
        base_generator.manual_seed(1024 + int(prompt_id) * 10 + i) 
        
        gen_img = pipe(
            add_template([f"Generate an image: {prompt_text}", "Generate an image."]), 
            guidance_scale=3.1, 
            generator=base_generator
        )
        gen_images.append(gen_img)
        
        # 4. 单张图片保存：以后缀区分，保存到 results 下
        img_filename = f"{prompt_id}_{i}.png"
        img_path = os.path.join(output_dir, img_filename)
        gen_img.save(img_path)
    
    # 5. 也可以选择将 4 张图拼成网格保存（可选保留）
    grid_image = create_image_grid(gen_images, 2, 2)
    grid_path = os.path.join(output_dir, f"{prompt_id}_grid.png")
    grid_image.save(grid_path)
    
    print(f"Finished saving images for {prompt_id} to {output_dir}/")

print("All tasks completed.")
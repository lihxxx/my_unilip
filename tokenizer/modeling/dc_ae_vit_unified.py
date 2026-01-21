"""Unified DC_AE_ViT model supporting all training stages.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from pathlib import Path
from copy import deepcopy

from omegaconf import OmegaConf
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel
from diffusers.models import AutoencoderDC

from modeling.modules.base_model import BaseModel


def pixel_shuffle(x, scale_factor=0.5):
    """Pixel shuffle operation for feature map reorganization."""
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
               int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.mlp = nn.Sequential(
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, channels, bias=True),
            nn.GELU(),
            nn.Linear(channels, channels, bias=True),
        )

    def forward(self, x):
        return x + self.mlp(x)


def dcae_decoder_forward_with_checkpoint(decoder, hidden_states: torch.Tensor) -> torch.Tensor:
    """Forward pass for DC-AE decoder with gradient checkpointing."""
    if decoder.in_shortcut:
        x = hidden_states.repeat_interleave(
            decoder.in_shortcut_repeats, dim=1, 
            output_size=hidden_states.shape[1] * decoder.in_shortcut_repeats
        )
        hidden_states = decoder.conv_in(hidden_states) + x
    else:
        hidden_states = decoder.conv_in(hidden_states)

    for up_block in reversed(decoder.up_blocks):
        hidden_states = torch.utils.checkpoint.checkpoint(
            up_block, hidden_states, use_reentrant=False
        )

    hidden_states = decoder.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
    hidden_states = decoder.conv_act(hidden_states)
    hidden_states = decoder.conv_out(hidden_states)
    return hidden_states


class DC_AE_ViT_Unified(BaseModel, PyTorchModelHubMixin):
    """
    Unified DC_AE_ViT model that supports all training stages through config.
    
    Config options:
        model.stage1_ckpt: str - Path to stage1 checkpoint (empty for stage1 training)
        model.use_gradient_checkpoint: bool - Use gradient checkpointing for encoder/decoder
        model.disable_drop_path: bool - Disable drop path in encoder (recommended for stage2)
        model.output_distill_feat: bool - Output distillation features (for stage2)
        dataset.preprocessing.crop_size: int - Output image size (224, 448, etc.)
    """
    
    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        self.embed_dim = config.model.embed_dim
        self.patch_size = config.model.patch_size
        
        # Get config options with defaults
        self.use_gradient_checkpoint = config.model.get("use_gradient_checkpoint", False)
        self.disable_drop_path = config.model.get("disable_drop_path", False)
        self.output_distill_feat = config.model.get("output_distill_feat", False)
        self.output_size = config.dataset.preprocessing.crop_size

        # Load MLLM model
        path = self.config.model.mllm_path
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True
        )
        self.encoder = model.vision_model
        vit_dim = model.vision_model.embeddings.patch_embedding.weight.shape[0]
        llm_hidden_size = model.config.llm_config.hidden_size

        # Build projection layers (from ViT to VAE decoder)
        down_blocks = []
        for i in range(3):
            down_blocks.append(ResBlock(llm_hidden_size))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.down_mlp = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        # Initialize weights
        self.apply(self._init_weights)
        
        # Reload pretrained weights (as they were reinitialized above)
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True
        )
        self.encoder = model.vision_model
        self.mlp1 = model.mlp1

        # Optionally disable drop path for stage2 training
        if self.disable_drop_path:
            for layer in self.encoder.encoder.layers:
                try:
                    layer.drop_path1.drop_prob = 0.0
                    layer.drop_path2.drop_prob = 0.0
                except:
                    continue
            print("Drop path disabled in encoder")

        # Load DC-AE decoder
        dc_ae = AutoencoderDC.from_pretrained(
            self.config.model.dc_ae_path, 
            torch_dtype=torch.float32
        )
        self.decoder = dc_ae.decoder
        for name, param in self.decoder.named_parameters():
            if len(param.data.shape) == 4:
                param.data = param.data.to(memory_format=torch.channels_last)

        # Load stage1 checkpoint if provided (for stage2 training)
        stage1_ckpt = self.config.model.get("stage1_ckpt", "")
        if stage1_ckpt:
            msg = self.load_state_dict(torch.load(stage1_ckpt), strict=False)
            print(f"Loaded stage1 checkpoint from {stage1_ckpt}")
            print(f"Missing keys: {msg.missing_keys}")
            print(f"Unexpected keys: {msg.unexpected_keys}")

        # Freeze encoder and mlp1 for stage1 training
        # (For stage2, all parameters are trainable but with different LR)
        freeze_encoder = self.config.model.get("freeze_encoder", True)
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.mlp1.requires_grad_(False)
        
        # Print trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        dict_config = OmegaConf.to_container(self.config)
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        """Encode input images to latent representation."""
        vit_embeds = self.encoder.embeddings(x)
        
        # Process through encoder layers
        for idx, encoder_layer in enumerate(self.encoder.encoder.layers):
            if self.use_gradient_checkpoint:
                vit_embeds = torch.utils.checkpoint.checkpoint(
                    encoder_layer, vit_embeds, use_reentrant=False
                )
            else:
                vit_embeds = encoder_layer(vit_embeds)
        
        # Remove CLS token and reshape
        vit_embeds = vit_embeds[:, 1:, :].contiguous().float()
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)

        # Store for distillation if needed
        distill_output = vit_embeds.clone() if self.output_distill_feat else None

        # Project to latent space
        for block in self.down_blocks:
            vit_embeds = block(vit_embeds)
        vit_embeds = self.down_mlp(vit_embeds)

        # Reshape to 2D feature map
        vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()
        b, c, hw = vit_embeds.shape
        vit_embeds = vit_embeds.view(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))

        z = vit_embeds

        # Return format depends on whether distillation is used
        if self.output_distill_feat:
            return z.float(), {'distill_feat': distill_output}
        else:
            return z.float(), z.float()

    def decode(self, z_quantized):
        """Decode latent representation to image."""
        if self.use_gradient_checkpoint:
            dec = dcae_decoder_forward_with_checkpoint(self.decoder, z_quantized)
        else:
            dec = self.decoder(z_quantized)
        
        dec = F.interpolate(dec, size=(self.output_size, self.output_size), 
                           mode='bilinear', align_corners=False)
        return dec

    def forward(self, x):
        """Forward pass: encode and decode."""
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, result_dict


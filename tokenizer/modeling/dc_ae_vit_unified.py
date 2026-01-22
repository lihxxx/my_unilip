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


class SemanticTransformerBlock(nn.Module):
    """Transformer block for Semantic Encoder/Decoder.
    
    Based on PS-VAE paper: The encoder and decoder share a symmetric design with 
    Transformer blocks inherited from the representation encoder.
    """
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class SemanticEncoder(nn.Module):
    """Semantic Encoder for PS-VAE.
    
    Maps high-dimensional representation features to a compact latent space.
    Uses Transformer blocks + MLP projection for dimensionality adjustment.
    """
    
    def __init__(self, input_dim, latent_dim, num_layers=3, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SemanticTransformerBlock(input_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # MLP projection to latent space (with mean and logvar for VAE)
        self.proj_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.proj_mean = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, latent_dim)
        )
        self.proj_logvar = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, latent_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, input_dim) - input features from representation encoder
        Returns:
            z: (B, N, latent_dim) - latent representation
            mean: (B, N, latent_dim) - mean of the latent distribution
            logvar: (B, N, latent_dim) - log variance of the latent distribution
        """
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Project to latent space
        x = self.proj_norm(x)
        mean = self.proj_mean(x)
        logvar = self.proj_logvar(x)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z, mean, logvar


class SemanticDecoder(nn.Module):
    """Semantic Decoder for PS-VAE.
    
    Reconstructs high-dimensional features from compact latent space.
    Symmetric design with Semantic Encoder.
    """
    
    def __init__(self, latent_dim, output_dim, num_layers=3, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # MLP projection from latent space
        self.proj_up = nn.Sequential(
            nn.Linear(latent_dim, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SemanticTransformerBlock(output_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(output_dim, eps=1e-6)
        
    def forward(self, z):
        """
        Args:
            z: (B, N, latent_dim) - latent representation
        Returns:
            x_recon: (B, N, output_dim) - reconstructed features
        """
        # Project up to feature dimension
        x = self.proj_up(z)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x_recon = self.final_norm(x)
        return x_recon


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

        # Check if semantic AE is enabled
        self.use_semantic_ae = config.model.get("use_semantic_ae", False)
        
        # Semantic AE config (PS-VAE style)
        # Based on paper: https://arxiv.org/pdf/2512.17909
        # S-VAE maps high-dimensional unconstrained feature space to compact latent space
        semantic_ae_config = config.model.get("semantic_ae", {})
        self.semantic_latent_dim = semantic_ae_config.get("latent_dim", 96)  # Default 96 channels as in PS-VAE
        self.semantic_num_layers = semantic_ae_config.get("num_layers", 3)   # 3 transformer blocks
        self.semantic_num_heads = semantic_ae_config.get("num_heads", 12)    # Match ViT heads
        self.semantic_mlp_ratio = semantic_ae_config.get("mlp_ratio", 4.0)
        
        if self.use_semantic_ae:
            # Build Semantic Encoder and Decoder (PS-VAE style)
            # The encoder and decoder share a symmetric design with Transformer blocks
            # inherited from the representation encoder and MLP projection layer
            self.semantic_encoder = SemanticEncoder(
                input_dim=llm_hidden_size,
                latent_dim=self.semantic_latent_dim,
                num_layers=self.semantic_num_layers,
                num_heads=self.semantic_num_heads,
                mlp_ratio=self.semantic_mlp_ratio
            )
            self.semantic_decoder = SemanticDecoder(
                latent_dim=self.semantic_latent_dim,
                output_dim=llm_hidden_size,
                num_layers=self.semantic_num_layers,
                num_heads=self.semantic_num_heads,
                mlp_ratio=self.semantic_mlp_ratio
            )
            
            # Build projection layers (from semantic latent to VAE decoder input)
            # Use ResBlocks and MLP similar to original design
            down_blocks = []
            for i in range(3):
                down_blocks.append(ResBlock(self.semantic_latent_dim))
            self.down_blocks = nn.ModuleList(down_blocks)
            self.down_mlp = nn.Sequential(
                nn.LayerNorm(self.semantic_latent_dim),
                nn.Linear(self.semantic_latent_dim, 32),
                nn.GELU(),
                nn.Linear(32, 32),
            )
            print(f"Semantic AE enabled with latent_dim={self.semantic_latent_dim}, "
                  f"num_layers={self.semantic_num_layers}, num_heads={self.semantic_num_heads}")
        else:
            # Original design: Direct projection from llm_hidden_size to 32
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
            print("Semantic AE disabled, using original projection design")
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
        """Encode input images to latent representation.
        
        If semantic AE is enabled, uses PS-VAE style encoding:
        1. ViT encoder -> high-dimensional features f_h
        2. Semantic Encoder -> compact latent z with KL regularization
        3. Semantic Decoder -> reconstructed features f_h'' (for semantic loss)
        4. ResBlocks + MLP -> final latent for pixel decoder
        """
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

        # Store original features for distillation if needed
        distill_output = vit_embeds.clone() if self.output_distill_feat else None
        
        # Result dictionary for extra outputs
        result_dict = {}
        
        if self.use_semantic_ae:
            # PS-VAE style encoding
            # f_h' = vit_embeds (high-dimensional features from representation encoder)
            f_h = vit_embeds  # (B, N, llm_hidden_size)
            
            # Semantic Encoder: f_h -> z (compact latent with KL regularization)
            z_semantic, z_mean, z_logvar = self.semantic_encoder(f_h)
            
            # Semantic Decoder: z -> f_h'' (reconstructed features)
            f_h_recon = self.semantic_decoder(z_semantic)
            
            # Store semantic AE outputs for loss computation
            result_dict['semantic_original'] = f_h          # Original features (for semantic loss)
            result_dict['semantic_reconstructed'] = f_h_recon  # Reconstructed features (for semantic loss)
            result_dict['semantic_mean'] = z_mean           # Mean (for KL loss)
            result_dict['semantic_logvar'] = z_logvar       # Log variance (for KL loss)
            result_dict['semantic_latent'] = z_semantic     # Latent (for visualization/debugging)
            
            # Project semantic latent to pixel decoder input
            latent_for_decoder = z_semantic
            for block in self.down_blocks:
                latent_for_decoder = block(latent_for_decoder)
            latent_for_decoder = self.down_mlp(latent_for_decoder)
        else:
            # Original design: direct projection
            latent_for_decoder = vit_embeds
            for block in self.down_blocks:
                latent_for_decoder = block(latent_for_decoder)
            latent_for_decoder = self.down_mlp(latent_for_decoder)

        # Reshape to 2D feature map for pixel decoder
        latent_for_decoder = latent_for_decoder.permute(0, 2, 1).contiguous()
        b, c, hw = latent_for_decoder.shape
        latent_for_decoder = latent_for_decoder.view(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))

        z = latent_for_decoder

        # Add distillation features if enabled
        if self.output_distill_feat:
            result_dict['distill_feat'] = distill_output

        # Return format for compatibility
        if result_dict:
            return z.float(), result_dict
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


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
import numpy as np

from omegaconf import OmegaConf
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel
from diffusers.models import AutoencoderDC

from modeling.modules.base_model import BaseModel


class TransformerBlock(nn.Module):
    """Transformer block for Semantic Encoder/Decoder.
    
    Based on PS-VAE paper: The encoder and decoder share a symmetric design with 
    Transformer blocks inherited from the representation encoder.
    """
    
    def __init__(self, hidden_size, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Self-attention with pre-norm
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # MLP with pre-norm
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
    def forward(self, x):
        # Self-attention with residual (pre-norm)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual (pre-norm)
        x = x + self.mlp(self.norm2(x))
        return x


class SemanticEncoder(nn.Module):
    """Semantic Encoder (S-Enc) for PS-VAE.
    
    Maps high-dimensional representation features to a compact latent space.
    Reference: vlvae_intervl_semae.py
    
    - Input: [B, N, D] high-dimensional features (e.g., D=2048 from InternVL3)
    - Output: [B, C, H, W] compact latent (e.g., C=latent_dim, H=W=8)
    """
    
    def __init__(self, input_dim, latent_dim, num_layers=3, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Transformer blocks inherited from representation encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(input_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # MLP projection layer for dimensionality adjustment
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, input_dim) - input features from representation encoder
        Returns:
            z: (B, latent_dim, H, W) - latent representation in spatial format
        """
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # MLP projection to latent space
        x = self.proj(x)  # [B, N, latent_dim]
        
        # Reshape to spatial format [B, N, C] -> [B, C, H, W]
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        z = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        return z


class SemanticDecoder(nn.Module):
    """Semantic Decoder (S-Dec) for PS-VAE.
    
    Reconstructs high-dimensional features from compact latent space.
    Symmetric design with Semantic Encoder.
    Reference: vlvae_intervl_semae.py
    
    - Input: [B, C, H, W] compact latent (e.g., C=latent_dim, H=W=8)
    - Output: [B, N, D] reconstructed features (e.g., D=2048)
    """
    
    def __init__(self, latent_dim, output_dim, num_layers=3, num_heads=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # MLP projection layer for dimensionality adjustment (symmetric with encoder)
        self.proj_in = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Transformer blocks inherited from representation encoder (symmetric with encoder)
        self.blocks = nn.ModuleList([
            TransformerBlock(output_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, z):
        """
        Args:
            z: (B, latent_dim, H, W) - latent representation in spatial format
        Returns:
            feat: (B, N, output_dim) - reconstructed features
        """
        B, C, H, W = z.shape
        # Reshape to sequence [B, C, H, W] -> [B, N, C]
        x = z.view(B, C, H * W).permute(0, 2, 1).contiguous()  # [B, N, C]
        
        # MLP projection to feature dim
        x = self.proj_in(x)  # [B, N, output_dim]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        return x


class CrossStreamAttention(nn.Module):
    """Cross-Stream Attention for Dual Stream Architecture.
    
    Enables information exchange between semantic and pixel streams.
    - Semantic stream can attend to pixel stream features
    - Pixel stream can attend to semantic stream features
    
    This helps:
    1. Semantic stream incorporates low-level details when needed
    2. Pixel stream incorporates high-level semantics for better reconstruction
    """
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Cross attention: semantic attends to pixel
        self.norm_sem = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_pix_for_sem = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn_sem = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross attention: pixel attends to semantic
        self.norm_pix = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_sem_for_pix = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn_pix = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # FFN for semantic stream after cross attention
        self.norm_ffn_sem = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_sem = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        # FFN for pixel stream after cross attention
        self.norm_ffn_pix = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_pix = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
    def forward(self, semantic_feat, pixel_feat):
        """
        Args:
            semantic_feat: (B, N, D) semantic stream features
            pixel_feat: (B, N, D) pixel stream features
        Returns:
            semantic_enhanced: (B, N, D) semantic features enhanced with pixel info
            pixel_enhanced: (B, N, D) pixel features enhanced with semantic info
        """
        # Cross attention: semantic attends to pixel (Q=semantic, K,V=pixel)
        sem_norm = self.norm_sem(semantic_feat)
        pix_for_sem = self.norm_pix_for_sem(pixel_feat)
        sem_cross, _ = self.cross_attn_sem(sem_norm, pix_for_sem, pix_for_sem)
        semantic_feat = semantic_feat + sem_cross
        
        # Cross attention: pixel attends to semantic (Q=pixel, K,V=semantic)
        pix_norm = self.norm_pix(pixel_feat)
        sem_for_pix = self.norm_sem_for_pix(semantic_feat)  # Use updated semantic
        pix_cross, _ = self.cross_attn_pix(pix_norm, sem_for_pix, sem_for_pix)
        pixel_feat = pixel_feat + pix_cross
        
        # FFN for both streams
        semantic_enhanced = semantic_feat + self.ffn_sem(self.norm_ffn_sem(semantic_feat))
        pixel_enhanced = pixel_feat + self.ffn_pix(self.norm_ffn_pix(pixel_feat))
        
        return semantic_enhanced, pixel_enhanced


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
        
        # Layer-wise distillation config (UniFlow style)
        # Reference: https://arxiv.org/pdf/2510.10575
        self.use_layerwise_distill = config.model.get("use_layerwise_distill", False)
        self.layerwise_distill_layers = config.model.get("layerwise_distill_layers", None)
        if self.use_layerwise_distill:
            print(f"Layer-wise distillation enabled, layers: {self.layerwise_distill_layers}")
        
        # Padding support for inference with different resolutions
        # Default is False, enable for inference when input resolution differs from training
        self.use_padding = config.model.get("use_padding", False)
        self.pad_info = None  # Will store padding info during encode for use in decode
        
        if self.use_padding:
            print("Padding mode enabled for inference with different resolutions")

        # Load MLLM model
        path = self.config.model.mllm_path
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.encoder = model.vision_model
        vit_dim = model.vision_model.embeddings.patch_embedding.weight.shape[0]
        llm_hidden_size = model.config.llm_config.hidden_size

        # Architecture type selection
        # Supported types:
        #   - "dual_stream_trans": Dual stream with transformer encoders and cross-stream attention
        #   - "dual_stream_simple": Dual stream with only down blocks (no transformers)
        #   - "dual_stream_pixeltrans": Dual stream with pixel transformer only (semantic is simple)
        #   - "semantic_ae": Semantic autoencoder (PS-VAE style)
        #   - "direct": Original direct projection design (default)
        self.arch_type = config.model.get("arch_type", "direct")
        
        # Validate architecture type
        valid_types = ["dual_stream_trans", "dual_stream_simple", "dual_stream_pixeltrans", "semantic_ae", "direct"]
        if self.arch_type not in valid_types:
            raise ValueError(f"Invalid arch_type: {self.arch_type}. Must be one of {valid_types}")
        
        print(f"Architecture type: {self.arch_type}")
        
        # Semantic AE config (PS-VAE style)
        # Based on paper: https://arxiv.org/pdf/2512.17909
        # S-VAE maps high-dimensional unconstrained feature space to compact latent space
        semantic_ae_config = config.model.get("semantic_ae", {})
        self.semantic_latent_dim = semantic_ae_config.get("latent_dim", 96)  # Default 96 channels as in PS-VAE
        
        # Encoder config
        encoder_config = semantic_ae_config.get("encoder", {})
        self.semantic_enc_num_layers = encoder_config.get("num_layers", 3)
        self.semantic_enc_num_heads = encoder_config.get("num_heads", 8)
        
        # Decoder config
        decoder_config = semantic_ae_config.get("decoder", {})
        self.semantic_dec_num_layers = decoder_config.get("num_layers", 3)
        self.semantic_dec_num_heads = decoder_config.get("num_heads", 8)
        
        # Dual stream config
        dual_stream_config = config.model.get("dual_stream", {})
        self.dual_stream_num_layers = dual_stream_config.get("num_layers", 3)
        self.dual_stream_num_heads = dual_stream_config.get("num_heads", 16)
        self.dual_stream_mlp_ratio = dual_stream_config.get("mlp_ratio", 4.0)
        self.dual_stream_dropout = dual_stream_config.get("dropout", 0.0)
        # Cross-stream interaction config
        self.use_cross_stream = dual_stream_config.get("use_cross_stream", False)
        self.cross_stream_num_heads = dual_stream_config.get("cross_stream_num_heads", 16)
        
        if self.arch_type == "dual_stream_trans":
            # ============ Dual Stream Architecture ============
            # Stream 1 (Semantic): vit_embeds -> TransformerEncoder -> semantic_feat (for distill_loss)
            #                      -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
            # Stream 2 (Pixel): vit_embeds -> pixel_transformer -> pixel_feat
            #                   -> down_blocks -> down_mlp -> 32-dim
            # Cross-Stream (optional): CrossStreamAttention for information exchange
            # Fusion: concat(32, 32) -> 64-dim -> fusion_layer -> 32-dim
            
            # Semantic stream: TransformerEncoder for semantic feature extraction
            semantic_encoder_layer = nn.TransformerEncoderLayer(
                d_model=llm_hidden_size,
                nhead=self.dual_stream_num_heads,
                dim_feedforward=int(llm_hidden_size * self.dual_stream_mlp_ratio),
                dropout=self.dual_stream_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.semantic_transformer = nn.TransformerEncoder(
                semantic_encoder_layer,
                num_layers=self.dual_stream_num_layers
            )
            
            # Pixel stream: TransformerEncoder for pixel feature extraction
            pixel_encoder_layer = nn.TransformerEncoderLayer(
                d_model=llm_hidden_size,
                nhead=self.dual_stream_num_heads,
                dim_feedforward=int(llm_hidden_size * self.dual_stream_mlp_ratio),
                dropout=self.dual_stream_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.pixel_transformer = nn.TransformerEncoder(
                pixel_encoder_layer,
                num_layers=self.dual_stream_num_layers
            )
            
            # Cross-stream attention (optional)
            if self.use_cross_stream:
                self.cross_stream_attention = CrossStreamAttention(
                    hidden_size=llm_hidden_size,
                    num_heads=self.cross_stream_num_heads,
                    dropout=self.dual_stream_dropout
                )
                print(f"  Cross-Stream Attention enabled with num_heads={self.cross_stream_num_heads}")
            
            # Semantic down blocks: llm_hidden_size -> 32
            semantic_down_blocks = []
            for i in range(3):
                semantic_down_blocks.append(ResBlock(llm_hidden_size))
            self.semantic_down_blocks = nn.ModuleList(semantic_down_blocks)
            self.semantic_down_mlp = nn.Sequential(
                nn.LayerNorm(llm_hidden_size),
                nn.Linear(llm_hidden_size, 32),
                nn.GELU(),
                nn.Linear(32, 32),
            )
            
            # Pixel stream: pixel_down_blocks -> pixel_down_mlp (llm_hidden_size -> 32)
            pixel_down_blocks = []
            for i in range(3):
                pixel_down_blocks.append(ResBlock(llm_hidden_size))
            self.pixel_down_blocks = nn.ModuleList(pixel_down_blocks)
            self.pixel_down_mlp = nn.Sequential(
                nn.LayerNorm(llm_hidden_size),
                nn.Linear(llm_hidden_size, 32),
                nn.GELU(),
                nn.Linear(32, 32),
            )
            
            # Fusion layer: 64 -> 32
            self.fusion_layer = nn.Sequential(
                nn.LayerNorm(64),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 32),
            )
            
            print(f"Dual Stream (Transformer) enabled with num_layers={self.dual_stream_num_layers}, "
                  f"num_heads={self.dual_stream_num_heads}, use_cross_stream={self.use_cross_stream}")
                  
        elif self.arch_type == "dual_stream_simple":
            # ============ Dual Stream Simple Architecture ============
            # Simplified dual stream without transformers, only down blocks
            # Stream 1 (Semantic): vit_embeds -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
            # Stream 2 (Pixel): vit_embeds -> pixel_down_blocks -> pixel_down_mlp -> 32-dim
            # Fusion: concat(32, 32) -> 64-dim -> fusion_layer -> 32-dim
            
            # Semantic stream: down_blocks -> down_mlp (llm_hidden_size -> 32)
            semantic_down_blocks = []
            for i in range(3):
                semantic_down_blocks.append(ResBlock(llm_hidden_size))
            self.semantic_down_blocks = nn.ModuleList(semantic_down_blocks)
            self.semantic_down_mlp = nn.Sequential(
                nn.LayerNorm(llm_hidden_size),
                nn.Linear(llm_hidden_size, 32),
                nn.GELU(),
                nn.Linear(32, 32),
            )
            
            # Pixel stream: down_blocks -> down_mlp (llm_hidden_size -> 32)
            pixel_down_blocks = []
            for i in range(3):
                pixel_down_blocks.append(ResBlock(llm_hidden_size))
            self.pixel_down_blocks = nn.ModuleList(pixel_down_blocks)
            self.pixel_down_mlp = nn.Sequential(
                nn.LayerNorm(llm_hidden_size),
                nn.Linear(llm_hidden_size, 32),
                nn.GELU(),
                nn.Linear(32, 32),
            )
            
            # Fusion layer: 64 -> 32
            self.fusion_layer = nn.Sequential(
                nn.LayerNorm(64),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 32),
            )
            
            print(f"Dual Stream (Simple) enabled - direct down blocks without transformers")
                  
        elif self.arch_type == "dual_stream_pixeltrans":
            # ============ Dual Stream Pixel Transformer Architecture ============
            # Semantic stream is simple (direct down blocks)
            # Pixel stream has transformer for better reconstruction capability
            # Stream 1 (Semantic): vit_embeds -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
            # Stream 2 (Pixel): vit_embeds -> pixel_transformer -> pixel_feat 
            #                   -> pixel_down_blocks -> pixel_down_mlp -> 32-dim
            # Fusion: concat(32, 32) -> 64-dim -> fusion_layer -> 32-dim
            
            # Pixel stream: TransformerEncoder for pixel feature extraction
            pixel_encoder_layer = nn.TransformerEncoderLayer(
                d_model=llm_hidden_size,
                nhead=self.dual_stream_num_heads,
                dim_feedforward=int(llm_hidden_size * self.dual_stream_mlp_ratio),
                dropout=self.dual_stream_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.pixel_transformer = nn.TransformerEncoder(
                pixel_encoder_layer,
                num_layers=self.dual_stream_num_layers
            )
            
            # Semantic stream: simple down_blocks (no transformer, preserve semantic)
            semantic_down_blocks = []
            for i in range(3):
                semantic_down_blocks.append(ResBlock(llm_hidden_size))
            self.semantic_down_blocks = nn.ModuleList(semantic_down_blocks)
            self.semantic_down_mlp = nn.Sequential(
                nn.LayerNorm(llm_hidden_size),
                nn.Linear(llm_hidden_size, 32),
                nn.GELU(),
                nn.Linear(32, 32),
            )
            
            # Pixel stream: down_blocks after transformer
            pixel_down_blocks = []
            for i in range(3):
                pixel_down_blocks.append(ResBlock(llm_hidden_size))
            self.pixel_down_blocks = nn.ModuleList(pixel_down_blocks)
            self.pixel_down_mlp = nn.Sequential(
                nn.LayerNorm(llm_hidden_size),
                nn.Linear(llm_hidden_size, 32),
                nn.GELU(),
                nn.Linear(32, 32),
            )
            
            # Fusion layer: 64 -> 32
            self.fusion_layer = nn.Sequential(
                nn.LayerNorm(64),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 32),
            )
            
            print(f"Dual Stream (PixelTrans) enabled with num_layers={self.dual_stream_num_layers}, "
                  f"num_heads={self.dual_stream_num_heads}")
            print(f"  Semantic stream: simple down blocks (preserve understanding)")
            print(f"  Pixel stream: transformer + down blocks (better reconstruction)")
                  
        elif self.arch_type == "semantic_ae":
            # Build Semantic Encoder and Decoder (PS-VAE style)
            # Reference: vlvae_intervl_semae.py
            # Encoder and Decoder can have different configurations
            self.semantic_encoder = SemanticEncoder(
                input_dim=llm_hidden_size,
                latent_dim=self.semantic_latent_dim,
                num_layers=self.semantic_enc_num_layers,
                num_heads=self.semantic_enc_num_heads
            )
            self.semantic_decoder = SemanticDecoder(
                latent_dim=self.semantic_latent_dim,
                output_dim=llm_hidden_size,
                num_layers=self.semantic_dec_num_layers,
                num_heads=self.semantic_dec_num_heads
            )
            
            # Build projection layers (from semantic latent to VAE decoder input)
            # semantic_latent_dim -> 32 for DC-AE decoder
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
            print(f"Semantic AE enabled with latent_dim={self.semantic_latent_dim}")
            print(f"  Encoder: num_layers={self.semantic_enc_num_layers}, num_heads={self.semantic_enc_num_heads}")
            print(f"  Decoder: num_layers={self.semantic_dec_num_layers}, num_heads={self.semantic_dec_num_heads}")
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
            # Use map_location="cpu" to avoid loading checkpoint directly to GPU,
            # which would cause OOM due to double memory usage
            msg = self.load_state_dict(torch.load(stage1_ckpt, map_location="cpu"), strict=False)
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

    def padding(self, tensor):
        """Pad tensor to multiple of 28 (ViT patch size) for inference.
        
        Args:
            tensor: Input tensor of shape (B, C, H, W), normalized by ImageNet stats.
            
        Returns:
            Padded tensor with dimensions as multiples of 28.
        """
        # Renormalize to [0,1]
        mean = self.config.dataset.preprocessing.normalize_mean
        std = self.config.dataset.preprocessing.normalize_std
        std = torch.tensor(std).to(tensor.device)
        mean = torch.tensor(mean).to(tensor.device)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        
        # Denormalize: inputs are normalized by ImageNet, convert to [0, 1]
        original_images = tensor * std + mean
        b, c, h, w = original_images.shape
        
        # Calculate target height and width (smallest multiple of 28 >= current size)
        new_h = math.ceil(h / 28) * 28
        new_w = math.ceil(w / 28) * 28
        
        # Calculate padding amounts
        pad_h = new_h - h
        pad_w = new_w - w
        
        # Calculate top/bottom and left/right padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Apply symmetric padding (reflect mode to avoid edge discontinuities)
        padded_images = F.pad(original_images, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        
        # Renormalize
        normalized_images = (padded_images - mean) / std
        self.pad_info = (pad_top, pad_bottom, pad_left, pad_right)
        return normalized_images
    
    def remove_padding(self, tensor):
        """Remove padding from the tensor after decoding.
        
        Args:
            tensor: Padded tensor of shape (B, C, H_padded, W_padded).
            
        Returns:
            Unpadded tensor with original spatial dimensions.
        """
        if self.pad_info is None:
            return tensor
            
        pad_top, pad_bottom, pad_left, pad_right = self.pad_info
        
        # Calculate original image region
        _, _, h_padded, w_padded = tensor.shape
        
        # Handle edge case where padding is 0
        h_end = h_padded - pad_bottom if pad_bottom > 0 else h_padded
        w_end = w_padded - pad_right if pad_right > 0 else w_padded
        
        unpadded_tensor = tensor[:, :, pad_top:h_end, pad_left:w_end]
        return unpadded_tensor
    
    def resize_down(self, tensor):
        """Resize down to 28/32 ratio after decoding.
        
        This is needed because the DC-AE decoder produces output at 32x scale,
        but ViT encoder uses 28x28 patches, so we need to scale down by 28/32.
        """
        _, _, h, w = tensor.shape
        
        # Calculate target size (rounding)
        target_h = round(h * 28 / 32)
        target_w = round(w * 28 / 32)
        
        return F.interpolate(
            tensor, 
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=True
        )

    def encode(self, x):
        """Encode input images to latent representation.
        
        If semantic AE is enabled, uses PS-VAE style encoding:
        1. ViT encoder -> high-dimensional features f_h
        2. Semantic Encoder -> compact latent z with KL regularization
        3. Semantic Decoder -> reconstructed features f_h'' (for semantic loss)
        4. ResBlocks + MLP -> final latent for pixel decoder
        
        If use_padding is enabled (for inference):
        - Pads input to multiple of 28 before encoding
        - Stores padding info for use in decode()
        
        If use_layerwise_distill is enabled:
        - Collects intermediate layer features for layer-wise distillation
        """
        # Apply padding if enabled (for inference with different resolutions)
        if self.use_padding:
            x = self.padding(x)
        
        vit_embeds = self.encoder.embeddings(x)
        
        # Determine which layers to collect for layer-wise distillation
        num_layers = len(self.encoder.encoder.layers)
        if self.use_layerwise_distill:
            if self.layerwise_distill_layers is not None:
                distill_layer_indices = set(self.layerwise_distill_layers)
            else:
                distill_layer_indices = set(range(num_layers))
        else:
            distill_layer_indices = set()
        
        # Collect layer-wise features for distillation
        layer_features = []
        
        # Process through encoder layers
        for idx, encoder_layer in enumerate(self.encoder.encoder.layers):
            if self.use_gradient_checkpoint:
                vit_embeds = torch.utils.checkpoint.checkpoint(
                    encoder_layer, vit_embeds, use_reentrant=False
                )
            else:
                vit_embeds = encoder_layer(vit_embeds)
            
            # Collect layer features for layer-wise distillation
            if idx in distill_layer_indices:
                # Extract features without CLS token
                layer_feat = vit_embeds[:, 1:, :].contiguous().float()
                layer_features.append(layer_feat)
        
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
        
        # Add layer-wise features for distillation
        if self.use_layerwise_distill and len(layer_features) > 0:
            result_dict['layer_features'] = layer_features
        
        if self.arch_type == "dual_stream_trans":
            # ============ Dual Stream Architecture ============
            # Stream 1 (Semantic): vit_embeds -> semantic_transformer -> semantic_feat
            # Stream 2 (Pixel): vit_embeds -> pixel_transformer -> pixel_feat
            # Cross-Stream (optional): CrossStreamAttention for information exchange
            # Then: semantic_feat -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
            #       pixel_feat -> down_blocks -> down_mlp -> 32-dim
            # Fusion: concat(32, 32) -> 64-dim -> fusion_layer -> 32-dim
            
            # Semantic stream: vit_embeds -> TransformerEncoder -> semantic_feat
            semantic_feat = self.semantic_transformer(vit_embeds)  # (B, N, llm_hidden_size)
            
            # Pixel stream: vit_embeds -> TransformerEncoder -> pixel_feat
            pixel_feat = self.pixel_transformer(vit_embeds)  # (B, N, llm_hidden_size)
            
            # Cross-stream attention (optional): information exchange between streams
            if self.use_cross_stream:
                semantic_feat, pixel_feat = self.cross_stream_attention(semantic_feat, pixel_feat)
            
            # Use semantic_feat for distillation loss (replaces vit_embeds)
            # v2
            # distill_output = semantic_feat.clone()
            
            # Store pixel_feat (before down) for pixel distillation if needed
            result_dict['pixel_feat'] = pixel_feat.clone()
            
            # Semantic stream: semantic_feat -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
            semantic_latent = semantic_feat
            for block in self.semantic_down_blocks:
                semantic_latent = block(semantic_latent)
            semantic_latent = self.semantic_down_mlp(semantic_latent)  # (B, N, 32)
            
            # Pixel stream: pixel_feat -> pixel_down_blocks -> pixel_down_mlp -> 32-dim
            pixel_latent = pixel_feat
            for block in self.pixel_down_blocks:
                pixel_latent = block(pixel_latent)
            pixel_latent = self.pixel_down_mlp(pixel_latent)  # (B, N, 32)
            
            # Store pixel_latent for pixel distillation loss
            result_dict['pixel_latent'] = pixel_latent.clone()
            
            # Fusion: concat(semantic_latent, pixel_latent) -> fusion_layer -> 32-dim
            fused_latent = torch.cat([semantic_latent, pixel_latent], dim=-1)  # (B, N, 64)
            latent_for_decoder = self.fusion_layer(fused_latent)  # (B, N, 32)
            
        elif self.arch_type == "dual_stream_simple":
            # ============ Dual Stream Simple Architecture ============
            # Simplified dual stream: directly apply down blocks without transformers
            # Stream 1 (Semantic): vit_embeds -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
            # Stream 2 (Pixel): vit_embeds -> pixel_down_blocks -> pixel_down_mlp -> 32-dim
            # Fusion: concat(32, 32) -> 64-dim -> fusion_layer -> 32-dim
            
            # Keep vit_embeds for distillation (no transformer processing)
            distill_output = vit_embeds.clone()
            
            # Semantic stream: vit_embeds -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
            semantic_latent = vit_embeds
            for block in self.semantic_down_blocks:
                semantic_latent = block(semantic_latent)
            semantic_latent = self.semantic_down_mlp(semantic_latent)  # (B, N, 32)
            
            # Pixel stream: vit_embeds -> pixel_down_blocks -> pixel_down_mlp -> 32-dim
            pixel_latent = vit_embeds
            for block in self.pixel_down_blocks:
                pixel_latent = block(pixel_latent)
            pixel_latent = self.pixel_down_mlp(pixel_latent)  # (B, N, 32)
            
            # Store pixel_latent for pixel distillation loss
            result_dict['pixel_latent'] = pixel_latent.clone()
            
            # Fusion: concat(semantic_latent, pixel_latent) -> fusion_layer -> 32-dim
            fused_latent = torch.cat([semantic_latent, pixel_latent], dim=-1)  # (B, N, 64)
            latent_for_decoder = self.fusion_layer(fused_latent)  # (B, N, 32)
            
        elif self.arch_type == "dual_stream_pixeltrans":
            # ============ Dual Stream Pixel Transformer Architecture ============
            # Semantic stream is simple (preserve understanding)
            # Pixel stream has transformer (better reconstruction)
            # Stream 1 (Semantic): vit_embeds -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
            # Stream 2 (Pixel): vit_embeds -> pixel_transformer -> pixel_feat
            #                   -> pixel_down_blocks -> pixel_down_mlp -> 32-dim
            # Fusion: concat(32, 32) -> 64-dim -> fusion_layer -> 32-dim
            
            # Keep vit_embeds for semantic distillation (no transformer processing)
            distill_output = vit_embeds.clone()
            
            # Semantic stream: vit_embeds -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
            # Keep simple to preserve semantic understanding
            semantic_latent = vit_embeds
            for block in self.semantic_down_blocks:
                semantic_latent = block(semantic_latent)
            semantic_latent = self.semantic_down_mlp(semantic_latent)  # (B, N, 32)
            
            # Pixel stream: vit_embeds -> pixel_transformer -> pixel_feat
            # Use transformer for better reconstruction capability
            pixel_feat = self.pixel_transformer(vit_embeds)  # (B, N, llm_hidden_size)
            
            # Store pixel_feat (before down) for pixel distillation if needed
            result_dict['pixel_feat'] = pixel_feat.clone()
            
            # Pixel stream: pixel_feat -> pixel_down_blocks -> pixel_down_mlp -> 32-dim
            pixel_latent = pixel_feat
            for block in self.pixel_down_blocks:
                pixel_latent = block(pixel_latent)
            pixel_latent = self.pixel_down_mlp(pixel_latent)  # (B, N, 32)
            
            # Store pixel_latent for pixel distillation loss
            result_dict['pixel_latent'] = pixel_latent.clone()
            
            # Fusion: concat(semantic_latent, pixel_latent) -> fusion_layer -> 32-dim
            fused_latent = torch.cat([semantic_latent, pixel_latent], dim=-1)  # (B, N, 64)
            latent_for_decoder = self.fusion_layer(fused_latent)  # (B, N, 32)
            
        elif self.arch_type == "semantic_ae":
            # PS-VAE style encoding (without KL, reference: vlvae_intervl_semae.py)
            # f_h = vit_embeds (high-dimensional features from trainable encoder)
            f_h = vit_embeds  # (B, N, llm_hidden_size)
            
            # Semantic Encoder: f_h -> z_spatial (compact latent in spatial format)
            z_spatial = self.semantic_encoder(f_h)  # (B, latent_dim, H, W)
            
            # Semantic Decoder: z_spatial -> f_h_recon (reconstructed features)
            # NOTE: f_h_recon should match the FROZEN ref encoder's features
            # The semantic reconstruction loss is computed in distill_loss.py using ref features
            f_h_recon = self.semantic_decoder(z_spatial)  # (B, N, llm_hidden_size)
            
            # Store semantic AE outputs for loss computation
            # semantic_reconstructed will be compared with frozen ref encoder features in distill_loss
            result_dict['semantic_reconstructed'] = f_h_recon  # Reconstructed features (for semantic loss)
            
            # Project semantic latent to pixel decoder input
            # Reshape z_spatial from (B, C, H, W) to (B, N, C) for down_blocks
            B, C, H, W = z_spatial.shape
            z_seq = z_spatial.view(B, C, H * W).permute(0, 2, 1).contiguous()  # (B, N, latent_dim)
            
            latent_for_decoder = z_seq
            for block in self.down_blocks:
                latent_for_decoder = block(latent_for_decoder)
            latent_for_decoder = self.down_mlp(latent_for_decoder)  # (B, N, 32)
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
        """Decode latent representation to image.
        
        If use_padding is enabled (for inference):
        - First resizes down by 28/32 ratio
        - Then removes padding to restore original dimensions
        """
        if self.use_gradient_checkpoint:
            dec = dcae_decoder_forward_with_checkpoint(self.decoder, z_quantized)
        else:
            dec = self.decoder(z_quantized)
        
        if self.use_padding:
            # For padding mode: resize down by 28/32 ratio, then remove padding
            dec = self.resize_down(dec)
            dec = self.remove_padding(dec)
        else:
            # Normal mode: simple interpolate to output size
            dec = F.interpolate(dec, size=(self.output_size, self.output_size), 
                               mode='bilinear', align_corners=False)
        return dec

    def forward(self, x):
        """Forward pass: encode and decode."""
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, result_dict


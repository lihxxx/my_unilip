"""This file contains distillation and semantic reconstruction loss modules.

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

References:
    - PS-VAE: https://arxiv.org/pdf/2512.17909
    - Semantic reconstruction loss combines L2 loss and cosine similarity loss
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from diffusers.models import AutoencoderDC
import math
import os
from typing import Dict, Tuple, Optional


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class SemanticReconstructionLoss(torch.nn.Module):
    """Semantic Reconstruction Loss for PS-VAE.
    
    Based on paper: https://arxiv.org/pdf/2512.17909
    
    The semantic reconstruction loss combines:
    1. L2 loss on features
    2. Cosine similarity loss on features
    
    This encourages the encoder to maintain fine-grained details during 
    the computation of strong semantic representations.
    """
    
    def __init__(self, l2_weight: float = 1.0, cosine_weight: float = 0.5):
        """Initialize SemanticReconstructionLoss.
        
        Args:
            l2_weight: Weight for L2 reconstruction loss.
            cosine_weight: Weight for cosine similarity loss.
        """
        super().__init__()
        self.l2_weight = l2_weight
        self.cosine_weight = cosine_weight
        
    def forward(self, 
                original_feat: torch.Tensor, 
                reconstructed_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute semantic reconstruction loss.
        
        Args:
            original_feat: Original features from representation encoder (B, N, D)
            reconstructed_feat: Reconstructed features from semantic decoder (B, N, D)
            
        Returns:
            total_loss: Combined semantic reconstruction loss
            loss_dict: Dictionary containing individual loss components
        """
        # L2 reconstruction loss
        l2_loss = F.mse_loss(reconstructed_feat, original_feat, reduction="mean")
        
        # Cosine similarity loss (1 - cosine_similarity)
        # Normalize features along the last dimension
        original_norm = F.normalize(original_feat, p=2, dim=-1)
        reconstructed_norm = F.normalize(reconstructed_feat, p=2, dim=-1)
        
        # Compute cosine similarity and convert to loss
        cosine_sim = (original_norm * reconstructed_norm).sum(dim=-1)  # (B, N)
        cosine_loss = (1.0 - cosine_sim).mean()
        
        # Combined loss
        total_loss = self.l2_weight * l2_loss + self.cosine_weight * cosine_loss
        
        loss_dict = {
            'semantic_l2_loss': l2_loss.detach(),
            'semantic_cosine_loss': cosine_loss.detach(),
            'semantic_total_loss': total_loss.detach()
        }
        
        return total_loss, loss_dict


class SemanticKLLoss(torch.nn.Module):
    """KL Divergence Loss for Semantic VAE.
    
    Based on PS-VAE paper: The latent is regularized by a KL divergence loss
    following standard VAE practice.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.
        
        KL(q(z|x) || p(z)) where p(z) = N(0, I)
        
        Args:
            mean: Mean of the latent distribution (B, N, D)
            logvar: Log variance of the latent distribution (B, N, D)
            
        Returns:
            KL divergence loss
        """
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()
        return kl_loss


class DistillLoss(torch.nn.Module):
    """Distillation loss with optional semantic reconstruction loss for PS-VAE.
    
    Based on paper: https://arxiv.org/pdf/2512.17909
    
    The semantic reconstruction loss ensures that the semantic decoder can reconstruct
    features that match the frozen reference encoder's output.
    
    For dual-stream mode, also supports pixel distillation loss using DC-AE encoder.
    """
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-1B",
                 use_semantic_loss: bool = False,
                 semantic_l2_weight: float = 1.0,
                 semantic_cosine_weight: float = 0.5,
                 use_pixel_distill: bool = False,
                 dc_ae_path: Optional[str] = None):
        """Initializes the Distill class.

        Args:
            model_name: A string, the path of the distillation loss model to use.
            use_semantic_loss: Whether to compute semantic reconstruction loss.
            semantic_l2_weight: Weight for L2 component in semantic loss.
            semantic_cosine_weight: Weight for cosine similarity component.
            use_pixel_distill: Whether to compute pixel distillation loss using DC-AE encoder.
            dc_ae_path: Path to DC-AE model for pixel distillation.
        """
        super().__init__()
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True)
        self.ref_vit = model.vision_model
        self.ref_mlp1 = model.mlp1

        for param in self.parameters():
            param.requires_grad = False
        
        # Semantic loss config
        self.use_semantic_loss = use_semantic_loss
        self.semantic_l2_weight = semantic_l2_weight
        self.semantic_cosine_weight = semantic_cosine_weight
        
        # Pixel distillation config
        self.use_pixel_distill = use_pixel_distill
        if use_pixel_distill and dc_ae_path:
            dc_ae = AutoencoderDC.from_pretrained(dc_ae_path, torch_dtype=torch.float32)
            self.ref_dc_ae_encoder = dc_ae.encoder
            for param in self.ref_dc_ae_encoder.parameters():
                param.requires_grad = False
            print(f"Loaded DC-AE encoder from {dc_ae_path} for pixel distillation")
    
    def _get_ref_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from frozen reference encoder.
        
        Args:
            x: Input image tensor (B, C, H, W), normalized to [0, 1].
            
        Returns:
            Reference features (B, N, D).
        """
        # x is in [0,1], need imgnet normalize
        std = torch.tensor([0.229, 0.224, 0.225]).to(x.device)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        x = (x - mean) / std
        
        # Always in eval mode
        self.eval()
        with torch.no_grad():
            vit_embeds = self.ref_vit.embeddings(x)
            for idx, encoder_layer in enumerate(self.ref_vit.encoder.layers):
                vit_embeds = encoder_layer(vit_embeds)
            vit_embeds = vit_embeds[:, 1:, :].contiguous().float()
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = pixel_shuffle(vit_embeds)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            vit_embeds = self.ref_mlp1(vit_embeds)
        
        return vit_embeds
    
    def _compute_semantic_loss(self, 
                                ref_feat: torch.Tensor, 
                                recon_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute semantic reconstruction loss.
        
        Args:
            ref_feat: Reference features from frozen encoder (B, N, D).
            recon_feat: Reconstructed features from semantic decoder (B, N, D).
            
        Returns:
            total_loss: Combined semantic reconstruction loss.
            loss_dict: Dictionary containing individual loss components.
        """
        # L2 reconstruction loss
        l2_loss = F.mse_loss(recon_feat, ref_feat, reduction="mean")
        
        # Cosine similarity loss (1 - cosine_similarity)
        ref_norm = F.normalize(ref_feat, p=2, dim=-1)
        recon_norm = F.normalize(recon_feat, p=2, dim=-1)
        cosine_sim = (ref_norm * recon_norm).sum(dim=-1)
        cosine_loss = (1.0 - cosine_sim).mean()
        
        # Combined loss
        total_loss = self.semantic_l2_weight * l2_loss + self.semantic_cosine_weight * cosine_loss
        
        loss_dict = {
            'semantic_l2_loss': l2_loss.detach(),
            'semantic_cosine_loss': cosine_loss.detach(),
        }
        
        return total_loss, loss_dict
    
    def _get_pixel_ref_features(self, x: torch.Tensor, target_size: int = 8) -> torch.Tensor:
        """Extract pixel features from frozen DC-AE encoder.
        
        Args:
            x: Input image tensor (B, C, H, W), normalized to [0, 1].
            target_size: Target spatial size to match pixel_latent (default 8 for 8x8=64 tokens).
            
        Returns:
            Reference pixel features (B, N, 32) reshaped from (B, 32, H', W').
        """
        # DC-AE encoder expects input in [-1, 1] range, convert from [0, 1]
        x_dcae = x * 2 - 1  # [0, 1] -> [-1, 1]
        
        self.eval()
        with torch.no_grad():
            # DC-AE encoder outputs (B, 32, H', W') where H'=W'=7 for 224x224 input
            pixel_feat = self.ref_dc_ae_encoder(x_dcae)  # (B, 32, 7, 7)
            
            # Interpolate to match pixel_latent spatial size (8x8 = 64 tokens)
            # pixel_latent comes from ViT (16x16) + pixel_shuffle -> 8x8
            b, c, h, w = pixel_feat.shape
            if h != target_size or w != target_size:
                pixel_feat = F.interpolate(
                    pixel_feat, 
                    size=(target_size, target_size), 
                    mode='bilinear', 
                    align_corners=False
                )  # (B, 32, 8, 8)
            
            # Reshape to (B, N, 32) to match pixel_latent format
            b, c, h, w = pixel_feat.shape
            pixel_feat = pixel_feat.permute(0, 2, 3, 1).contiguous()  # (B, 8, 8, 32)
            pixel_feat = pixel_feat.view(b, h * w, c)  # (B, 64, 32)
        
        return pixel_feat
    
    def _compute_pixel_distill_loss(self, 
                                     ref_pixel_feat: torch.Tensor, 
                                     pixel_latent: torch.Tensor) -> torch.Tensor:
        """Compute pixel distillation loss.
        
        Args:
            ref_pixel_feat: Reference pixel features from frozen DC-AE encoder (B, N, 32).
            pixel_latent: Pixel latent from dual-stream model (B, N, 32).
            
        Returns:
            pixel_distill_loss: MSE loss between ref and predicted pixel features.
        """
        pixel_distill_loss = F.mse_loss(pixel_latent, ref_pixel_feat, reduction="mean")
        return pixel_distill_loss
    
    def forward(self, x: torch.Tensor, out_feat: torch.Tensor, 
                semantic_reconstructed: torch.Tensor = None,
                pixel_latent: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the distillation loss and optional semantic/pixel reconstruction loss.

        Args:
            x: Input image tensor (B, C, H, W), normalized to [0, 1].
            out_feat: Output features from the trainable encoder (B, N, D).
            semantic_reconstructed: Reconstructed features from semantic decoder (B, N, D).
                                   Only used when use_semantic_loss=True.
            pixel_latent: Pixel latent from dual-stream model (B, N, 32).
                         Only used when use_pixel_distill=True.

        Returns:
            distill_loss: The semantic distillation loss.
            semantic_recon_loss: The semantic reconstruction loss (0 if not enabled).
            pixel_distill_loss: The pixel distillation loss (0 if not enabled).
            loss_dict: Dictionary containing all loss components.
        """
        # Get reference features from frozen encoder
        ref_feat = self._get_ref_features(x)
        
        # Compute distillation loss (trainable encoder vs frozen encoder)
        distill_loss = F.mse_loss(out_feat, ref_feat, reduction="mean")
        
        loss_dict = {
            'distill_loss_raw': distill_loss.detach(),
        }
        
        # Compute semantic reconstruction loss if enabled
        semantic_recon_loss = torch.tensor(0.0).to(x.device)
        if self.use_semantic_loss and semantic_reconstructed is not None:
            semantic_recon_loss, semantic_loss_dict = self._compute_semantic_loss(
                ref_feat, semantic_reconstructed
            )
            loss_dict.update(semantic_loss_dict)
        
        # Compute pixel distillation loss if enabled
        pixel_distill_loss = torch.tensor(0.0).to(x.device)
        if self.use_pixel_distill and pixel_latent is not None:
            ref_pixel_feat = self._get_pixel_ref_features(x)
            pixel_distill_loss = self._compute_pixel_distill_loss(ref_pixel_feat, pixel_latent)

        return distill_loss, semantic_recon_loss, pixel_distill_loss, loss_dict

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
    - UniFlow: Layer-wise Adaptive Self-Distillation (https://arxiv.org/pdf/2510.10575)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from diffusers.models import AutoencoderDC
import math
import os
from typing import Dict, Tuple, Optional, List


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
    
    Supports Layer-wise Adaptive Self-Distillation (UniFlow):
    - Reference: https://arxiv.org/pdf/2510.10575
    - Distills features from all encoder layers with adaptive weights
    - Deeper layers get higher base weights (preserve semantic capability)
    - Poorly aligned layers get additional penalty weight
    """
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-1B",
                 use_semantic_loss: bool = False,
                 semantic_l2_weight: float = 1.0,
                 semantic_cosine_weight: float = 0.5,
                 use_pixel_distill: bool = False,
                 dc_ae_path: Optional[str] = None,
                 use_layerwise_distill: bool = False,
                 layerwise_beta: float = 2.0,
                 layerwise_distill_layers: Optional[List[int]] = None,
                 distill_loss_type: str = "mse",
                 distill_cosine_weight: float = 1.0):
        """Initializes the Distill class.

        Args:
            model_name: A string, the path of the distillation loss model to use.
            use_semantic_loss: Whether to compute semantic reconstruction loss.
            semantic_l2_weight: Weight for L2 component in semantic loss.
            semantic_cosine_weight: Weight for cosine similarity component.
            use_pixel_distill: Whether to compute pixel distillation loss using DC-AE encoder.
            dc_ae_path: Path to DC-AE model for pixel distillation.
            use_layerwise_distill: Whether to use layer-wise adaptive self-distillation (UniFlow style).
            layerwise_beta: Temperature parameter for adaptive weight calculation. 
                           Higher beta emphasizes poorly aligned layers more.
            layerwise_distill_layers: List of layer indices to use for distillation.
                                     If None, uses all layers.
            distill_loss_type: Type of distillation loss. Options:
                "mse" - Mean Squared Error loss (default)
                "cosine" - Cosine similarity loss (1 - cos_sim)
                "mse+cosine" - Weighted sum of MSE and cosine loss
            distill_cosine_weight: Weight for cosine loss when using "mse+cosine" mode.
                                  MSE weight is implicitly 1.0.
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
        
        # Get number of encoder layers
        self.num_layers = len(self.ref_vit.encoder.layers)

        for param in self.parameters():
            param.requires_grad = False
        
        # Distillation loss type config
        assert distill_loss_type in ("mse", "cosine", "mse+cosine"), \
            f"Unsupported distill_loss_type: {distill_loss_type}. Use 'mse', 'cosine', or 'mse+cosine'."
        self.distill_loss_type = distill_loss_type
        self.distill_cosine_weight = distill_cosine_weight
        print(f"Distillation loss type: {distill_loss_type}" +
              (f" (cosine_weight={distill_cosine_weight})" if distill_loss_type == "mse+cosine" else ""))
        
        # Semantic loss config
        self.use_semantic_loss = use_semantic_loss
        self.semantic_l2_weight = semantic_l2_weight
        self.semantic_cosine_weight = semantic_cosine_weight
        
        # Layer-wise distillation config (UniFlow style)
        self.use_layerwise_distill = use_layerwise_distill
        self.layerwise_beta = layerwise_beta
        
        # Determine which layers to use for distillation
        if layerwise_distill_layers is not None:
            self.distill_layers = layerwise_distill_layers
        else:
            # Use all layers by default
            self.distill_layers = list(range(self.num_layers))
        
        if use_layerwise_distill:
            print(f"Layer-wise Adaptive Self-Distillation enabled:")
            print(f"  - Total encoder layers: {self.num_layers}")
            print(f"  - Distillation layers: {self.distill_layers}")
            print(f"  - Beta (temperature): {layerwise_beta}")
        
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
    
    def _get_layerwise_ref_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract layer-wise features from frozen reference encoder.
        
        Reference: UniFlow (https://arxiv.org/pdf/2510.10575)
        
        Args:
            x: Input image tensor (B, C, H, W), normalized to [0, 1].
            
        Returns:
            List of reference features for each layer (B, N, D).
        """
        # x is in [0,1], need imgnet normalize
        std = torch.tensor([0.229, 0.224, 0.225]).to(x.device)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        x = (x - mean) / std
        
        # Always in eval mode
        self.eval()
        layer_features = []
        with torch.no_grad():
            vit_embeds = self.ref_vit.embeddings(x)
            for idx, encoder_layer in enumerate(self.ref_vit.encoder.layers):
                vit_embeds = encoder_layer(vit_embeds)
                if idx in self.distill_layers:
                    # Extract features without CLS token
                    layer_feat = vit_embeds[:, 1:, :].contiguous().float()
                    layer_features.append(layer_feat)
        
        return layer_features
    
    def _compute_layerwise_distill_loss(
        self, 
        student_features: List[torch.Tensor], 
        teacher_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute layer-wise adaptive self-distillation loss.
        
        Reference: UniFlow (https://arxiv.org/pdf/2510.10575)
        
        The loss is computed as:
        L_dist = sum_{l} w_l * (1 - cos_sim(H_U^l, H_T^l))
        
        where w_l = softmax(w_l^base * exp(beta * alpha_l))
        - w_l^base = l / L (deeper layers get higher base weight)
        - alpha_l = 1 - mean(cos_sim) (alignment penalty)
        - beta: temperature parameter
        
        Args:
            student_features: List of student encoder features [(B, N, D), ...]
            teacher_features: List of teacher encoder features [(B, N, D), ...]
            
        Returns:
            layerwise_distill_loss: Combined layer-wise distillation loss
            loss_dict: Dictionary containing layer-wise loss components
        """
        assert len(student_features) == len(teacher_features), \
            f"Student has {len(student_features)} layers, teacher has {len(teacher_features)} layers"
        
        num_distill_layers = len(self.distill_layers)
        device = student_features[0].device
        
        # Compute per-layer cosine distance and alignment penalty
        layer_cos_distances = []  # 1 - cos_sim
        layer_alphas = []  # alignment penalty
        
        for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
            # Normalize features
            student_norm = F.normalize(student_feat, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_feat, p=2, dim=-1)
            
            # Compute cosine similarity per token
            cos_sim = (student_norm * teacher_norm).sum(dim=-1)  # (B, N)
            
            # Cosine distance: 1 - cos_sim
            cos_distance = (1.0 - cos_sim).mean()  # scalar
            layer_cos_distances.append(cos_distance)
            
            # Alignment penalty: mean of (1 - cos_sim)
            alpha_l = cos_distance.detach()
            layer_alphas.append(alpha_l)
        
        # Compute adaptive weights
        # w_l^base = (l + 1) / L (1-indexed, deeper layers get higher weight)
        base_weights = []
        for i, layer_idx in enumerate(self.distill_layers):
            # Use actual layer index for base weight calculation
            w_base = (layer_idx + 1) / self.num_layers
            base_weights.append(w_base)
        
        base_weights = torch.tensor(base_weights, device=device)
        layer_alphas_tensor = torch.stack(layer_alphas)
        
        # Compute adaptive weights: w_l = softmax(w_l^base * exp(beta * alpha_l))
        # This emphasizes both deeper layers and poorly aligned layers
        weighted_scores = base_weights * torch.exp(self.layerwise_beta * layer_alphas_tensor)
        adaptive_weights = F.softmax(weighted_scores, dim=0)
        
        # Compute weighted sum of layer losses
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        for i, (cos_dist, weight) in enumerate(zip(layer_cos_distances, adaptive_weights)):
            weighted_loss = weight * cos_dist
            total_loss = total_loss + weighted_loss
            
            layer_idx = self.distill_layers[i]
            loss_dict[f'layerwise_distill_layer{layer_idx}'] = cos_dist.detach()
            loss_dict[f'layerwise_weight_layer{layer_idx}'] = weight.detach()
        
        loss_dict['layerwise_distill_total'] = total_loss.detach()
        
        return total_loss, loss_dict
    
    def _compute_distill_loss(
        self,
        out_feat: torch.Tensor,
        ref_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute distillation loss based on configured loss type.
        
        Args:
            out_feat: Output features from the trainable encoder (B, N, D).
            ref_feat: Reference features from frozen encoder (B, N, D).
            
        Returns:
            distill_loss: The computed distillation loss.
            loss_dict: Dictionary containing loss components for logging.
        """
        loss_dict = {}
        
        if self.distill_loss_type == "mse":
            distill_loss = F.mse_loss(out_feat, ref_feat, reduction="mean")
            
        elif self.distill_loss_type == "cosine":
            cosine_sim = F.cosine_similarity(out_feat, ref_feat, dim=-1)  # (B, N)
            distill_loss = (1.0 - cosine_sim).mean()
            
        elif self.distill_loss_type == "mse+cosine":
            mse_loss = F.mse_loss(out_feat, ref_feat, reduction="mean")
            cosine_sim = F.cosine_similarity(out_feat, ref_feat, dim=-1)
            cosine_loss = (1.0 - cosine_sim).mean()
            distill_loss = mse_loss + self.distill_cosine_weight * cosine_loss
            loss_dict['distill_mse_component'] = mse_loss.detach()
            loss_dict['distill_cosine_component'] = cosine_loss.detach()
        
        return distill_loss, loss_dict
    
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
    
    def _get_pixel_ref_features(self, x: torch.Tensor, target_size: int = None) -> torch.Tensor:
        """Extract pixel features from frozen DC-AE encoder.
        
        Args:
            x: Input image tensor (B, C, H, W), normalized to [0, 1].
            target_size: Target spatial size to match pixel_latent. If None, will be
                         computed based on input image size (H/14/2 for patch_size=14 with pixel_shuffle).
            
        Returns:
            Reference pixel features (B, N, 32) reshaped from (B, 32, H', W').
        """
        # DC-AE encoder expects input in [-1, 1] range, convert from [0, 1]
        x_dcae = x * 2 - 1  # [0, 1] -> [-1, 1]
        
        # Calculate target_size based on input image size if not provided
        # For ViT with patch_size=14 and pixel_shuffle(scale=0.5):
        # 224x224 -> 16x16 patches -> 8x8 after pixel_shuffle -> target_size=8
        # 448x448 -> 32x32 patches -> 16x16 after pixel_shuffle -> target_size=16
        if target_size is None:
            patch_size = 14  # InternVL3 uses patch_size=14
            img_size = x.shape[-1]  # H or W
            target_size = img_size // patch_size // 2  # div by 2 due to pixel_shuffle
        
        self.eval()
        with torch.no_grad():
            # DC-AE encoder outputs (B, 32, H', W') where H'=W'=7 for 224x224 input
            pixel_feat = self.ref_dc_ae_encoder(x_dcae)  # (B, 32, 7, 7) for 224x224
            
            # Interpolate to match pixel_latent spatial size
            # pixel_latent comes from ViT + pixel_shuffle
            b, c, h, w = pixel_feat.shape
            if h != target_size or w != target_size:
                pixel_feat = F.interpolate(
                    pixel_feat, 
                    size=(target_size, target_size), 
                    mode='bilinear', 
                    align_corners=False
                )  # (B, 32, target_size, target_size)
            
            # Reshape to (B, N, 32) to match pixel_latent format
            b, c, h, w = pixel_feat.shape
            pixel_feat = pixel_feat.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 32)
            pixel_feat = pixel_feat.view(b, h * w, c)  # (B, N, 32)
        
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
                pixel_latent: torch.Tensor = None,
                student_layer_features: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the distillation loss and optional semantic/pixel reconstruction loss.

        Args:
            x: Input image tensor (B, C, H, W), normalized to [0, 1].
            out_feat: Output features from the trainable encoder (B, N, D).
                     In dual-stream mode, this is semantic_feat (after semantic_transformer).
            semantic_reconstructed: Reconstructed features from semantic decoder (B, N, D).
                                   Only used when use_semantic_loss=True.
            pixel_latent: Pixel latent from dual-stream model (B, N, 32).
                         Only used when use_pixel_distill=True.
            student_layer_features: List of layer-wise features from trainable encoder (ViT).
                                   Only used when use_layerwise_distill=True.
                                   Note: This is independent of semantic_transformer output.

        Returns:
            distill_loss: The semantic distillation loss (semantic_feat vs frozen encoder).
            layerwise_distill_loss: The layer-wise distillation loss (ViT layers vs frozen encoder).
            semantic_recon_loss: The semantic reconstruction loss (0 if not enabled).
            pixel_distill_loss: The pixel distillation loss (0 if not enabled).
            loss_dict: Dictionary containing all loss components.
        """
        loss_dict = {}
        
        # Get reference features from frozen encoder (always needed for distill_loss)
        ref_feat = self._get_ref_features(x)
        
        # Compute standard distillation loss (semantic_feat/out_feat vs frozen encoder)
        # In dual-stream mode: out_feat is semantic_feat (after semantic_transformer)
        # This loss ensures semantic_transformer output aligns with frozen encoder
        distill_loss, distill_type_dict = self._compute_distill_loss(out_feat, ref_feat)
        loss_dict['distill_loss_raw'] = distill_loss.detach()
        loss_dict.update(distill_type_dict)
        
        # Compute layer-wise distillation loss if enabled (independent of above)
        # This loss ensures ViT encoder layers align with frozen encoder layers
        # It does NOT conflict with distill_loss because:
        # - layerwise_distill: ViT encoder intermediate layers vs frozen ViT layers
        # - distill_loss: semantic_transformer output vs frozen encoder final output
        layerwise_distill_loss = torch.tensor(0.0).to(x.device)
        if self.use_layerwise_distill and student_layer_features is not None:
            # Get layer-wise reference features from frozen encoder
            teacher_layer_features = self._get_layerwise_ref_features(x)
            
            # Compute layer-wise adaptive distillation loss
            layerwise_distill_loss, layerwise_loss_dict = self._compute_layerwise_distill_loss(
                student_layer_features, teacher_layer_features
            )
            loss_dict.update(layerwise_loss_dict)
        
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

        return distill_loss, layerwise_distill_loss, semantic_recon_loss, pixel_distill_loss, loss_dict
    
    def get_dc_ae_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get DC-AE encoder features for VF Loss computation.
        
        This is a public interface to access DC-AE features from external modules.
        
        Args:
            x: Input image tensor (B, C, H, W), normalized to [0, 1].
            
        Returns:
            DC-AE features (B, N, 32) if DC-AE encoder is available, None otherwise.
        """
        if not self.use_pixel_distill or not hasattr(self, 'ref_dc_ae_encoder'):
            return None
        return self._get_pixel_ref_features(x)
    
    @property
    def dc_ae_encoder(self):
        """Property to check if DC-AE encoder is available."""
        if hasattr(self, 'ref_dc_ae_encoder'):
            return self.ref_dc_ae_encoder
        return None

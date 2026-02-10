"""Unified training loss implementation.

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

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
    https://github.com/svg-project/SVG (VF Loss and adaptive weight)
"""
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange

from .perceptual_loss import PerceptualLoss
from .discriminator import NLayerDiscriminator
from .distill_loss import DistillLoss, SemanticKLLoss


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits."""
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


def compute_vf_loss(
    z: torch.Tensor, 
    aux_feature: torch.Tensor,
    distmat_margin: float = 0.0,
    cos_margin: float = 0.0,
    distmat_weight: float = 1.0,
    cos_weight: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    Compute Visual Feature (VF) Loss for pixel stream.
    Reference: SVG (https://github.com/svg-project/SVG)
    
    VF Loss consists of two parts:
    1. Distance Matrix Loss: Preserves pairwise similarity structure between patches
    2. Cosine Similarity Loss: Direct feature alignment at each position
    
    Args:
        z: Latent features (B, N, C) or (B, C, H, W)
        aux_feature: Reference features (B, N, C) or (B, C, H, W)
        distmat_margin: Margin for distance matrix loss
        cos_margin: Margin for cosine similarity loss
        distmat_weight: Weight for distance matrix loss
        cos_weight: Weight for cosine similarity loss
        
    Returns:
        vf_loss: Combined VF loss
        loss_dict: Dictionary containing individual loss components
    """
    # Handle different input shapes
    if len(z.shape) == 4:
        # (B, C, H, W) -> (B, C, H*W)
        z_flat = rearrange(z, 'b c h w -> b c (h w)')
        aux_flat = rearrange(aux_feature, 'b c h w -> b c (h w)')
    else:
        # (B, N, C) -> (B, C, N)
        z_flat = z.permute(0, 2, 1).contiguous()
        aux_flat = aux_feature.permute(0, 2, 1).contiguous()
    
    # Normalize along channel dimension
    z_norm = F.normalize(z_flat, dim=1)
    aux_norm = F.normalize(aux_flat, dim=1)
    
    # 1. Distance Matrix Loss - preserves pairwise similarity structure
    # Compute patch-to-patch similarity matrices
    z_cos_sim = torch.einsum('bci,bcj->bij', z_norm, z_norm)  # (B, N, N)
    aux_cos_sim = torch.einsum('bci,bcj->bij', aux_norm, aux_norm)  # (B, N, N)
    
    diff = torch.abs(z_cos_sim - aux_cos_sim)
    vf_loss_distmat = F.relu(diff - distmat_margin).mean()
    
    # 2. Cosine Similarity Loss - direct position-wise alignment
    if len(z.shape) == 4:
        cos_sim = F.cosine_similarity(aux_feature, z, dim=1)  # (B, H, W)
    else:
        cos_sim = F.cosine_similarity(aux_feature, z, dim=-1)  # (B, N)
    vf_loss_cos = F.relu(1.0 - cos_margin - cos_sim).mean()
    
    # Combined VF loss
    vf_loss = distmat_weight * vf_loss_distmat + cos_weight * vf_loss_cos
    
    loss_dict = {
        'vf_loss_distmat': vf_loss_distmat.detach(),
        'vf_loss_cos': vf_loss_cos.detach(),
        'vf_loss_total': vf_loss.detach()
    }
    
    return vf_loss, loss_dict


class ReconstructionLoss_Unified(torch.nn.Module):
    """
    Unified reconstruction loss supporting all training stages.
    
    Config options:
        losses.distill_loss: str - Path to distill model (empty to disable)
        losses.distill_weight: float - Weight for distillation loss (0 to disable)
        losses.discriminator_start: int - Step to start discriminator training
        losses.perceptual_loss: str - Perceptual loss config string
        losses.perceptual_weight: float - Weight for perceptual loss
        losses.reconstruction_loss: str - "l1" or "l2"
        losses.reconstruction_weight: float - Weight for reconstruction loss
        losses.kl_weight: float - Weight for KL loss (for VAE mode)
        losses.semantic_recon_weight: float - Weight for semantic reconstruction loss (PS-VAE)
        losses.semantic_kl_weight: float - Weight for semantic KL loss (PS-VAE)
        losses.pixel_distill_weight: float - Weight for pixel distillation loss (dual-stream)
        losses.vf_loss_weight: float - Weight for VF loss (pixel stream structure preservation)
        losses.use_adaptive_weight: bool - Enable adaptive weight for losses
    """
    
    def __init__(self, config):
        """Initializes the unified losses module."""
        super().__init__()
        loss_config = config.losses
        
        # Discriminator
        self.discriminator = NLayerDiscriminator()
        
        # Basic losses
        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = PerceptualLoss(loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start
        
        # Discriminator config
        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        # VAE mode config
        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode == "vae":
            self.kl_weight = loss_config.get("kl_weight", 1e-6)
            logvar_init = loss_config.get("logvar_init", 0.0)
            self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=False)
        
        # Architecture type selection
        self.arch_type = config.model.get("arch_type", "direct")
        
        # Semantic AE config (PS-VAE, without KL)
        # Based on paper: https://arxiv.org/pdf/2512.17909
        # Reference: vlvae_intervl_semae.py (simplified version without KL)
        self.semantic_recon_weight = loss_config.get("semantic_recon_weight", 1.0)
        
        # Dual stream config
        self.pixel_distill_weight = loss_config.get("pixel_distill_weight", 0.0)
        
        # VF Loss config (for pixel stream structure preservation)
        # Reference: SVG (https://github.com/svg-project/SVG)
        self.vf_loss_weight = loss_config.get("vf_loss_weight", 0.0)
        self.vf_distmat_weight = loss_config.get("vf_distmat_weight", 1.0)
        self.vf_cos_weight = loss_config.get("vf_cos_weight", 1.0)
        self.vf_distmat_margin = loss_config.get("vf_distmat_margin", 0.0)
        self.vf_cos_margin = loss_config.get("vf_cos_margin", 0.0)
        # VF loss can be used with all dual stream architectures
        self.use_vf_loss = (self.arch_type in ["dual_stream_trans", "dual_stream_simple", "dual_stream_pixeltrans"]) and self.vf_loss_weight > 0.0
        if self.use_vf_loss:
            print(f"VF Loss enabled with weight {self.vf_loss_weight}, "
                  f"distmat_weight={self.vf_distmat_weight}, cos_weight={self.vf_cos_weight}")
        
        # Adaptive weight config (reference: SVG)
        # Dynamically adjust loss weights based on gradient norms
        self.use_adaptive_weight = loss_config.get("use_adaptive_weight", False)
        self.adaptive_vf_weight = loss_config.get("adaptive_vf_weight", False)
        self.adaptive_distill_weight = loss_config.get("adaptive_distill_weight", False)
        self.adaptive_pixel_distill_weight = loss_config.get("adaptive_pixel_distill_weight", False)
        self.adaptive_weight_max = loss_config.get("adaptive_weight_max", 1e4)
        if self.use_adaptive_weight:
            print(f"Adaptive weight enabled: vf={self.adaptive_vf_weight}, "
                  f"distill={self.adaptive_distill_weight}, pixel_distill={self.adaptive_pixel_distill_weight}")
        
        # Layer-wise distillation config (UniFlow style)
        # Reference: https://arxiv.org/pdf/2510.10575
        self.use_layerwise_distill = config.model.get("use_layerwise_distill", False)
        self.layerwise_beta = loss_config.get("layerwise_distill_beta", 2.0)
        self.layerwise_distill_layers = config.model.get("layerwise_distill_layers", None)
        self.layerwise_distill_weight = loss_config.get("layerwise_distill_weight", 1.0)
        
        # Distillation loss (optional, for stage2)
        # When semantic AE is enabled, distill_loss also computes semantic reconstruction loss
        # When dual stream is enabled, distill_loss also computes pixel distillation loss
        self.use_distill = False
        distill_loss_path = loss_config.get("distill_loss", "")
        self.distill_weight = loss_config.get("distill_weight", 0.0)
        if distill_loss_path and self.distill_weight > 0.0:
            self.use_distill = True
            semantic_l2_weight = loss_config.get("semantic_l2_weight", 1.0)
            semantic_cosine_weight = loss_config.get("semantic_cosine_weight", 0.5)
            
            # Get DC-AE path for pixel distillation in dual-stream mode (all dual stream types)
            dc_ae_path = config.model.get("dc_ae_path", None) if self.arch_type in ["dual_stream_trans", "dual_stream_simple", "dual_stream_pixeltrans"] else None
            use_pixel_distill = (self.arch_type in ["dual_stream_trans", "dual_stream_simple", "dual_stream_pixeltrans"]) and self.pixel_distill_weight > 0.0
            
            self.distill_loss = DistillLoss(
                distill_loss_path,
                use_semantic_loss=(self.arch_type == "semantic_ae"),
                semantic_l2_weight=semantic_l2_weight,
                semantic_cosine_weight=semantic_cosine_weight,
                use_pixel_distill=use_pixel_distill,
                dc_ae_path=dc_ae_path,
                use_layerwise_distill=self.use_layerwise_distill,
                layerwise_beta=self.layerwise_beta,
                layerwise_distill_layers=self.layerwise_distill_layers
            ).eval()
            print(f"Distillation loss enabled with weight {self.distill_weight}")
            if self.use_layerwise_distill:
                print(f"Layer-wise Adaptive Self-Distillation enabled (UniFlow style)")
                print(f"  - Beta: {self.layerwise_beta}")
                print(f"  - Layers: {self.layerwise_distill_layers if self.layerwise_distill_layers else 'all'}")
            if self.arch_type == "semantic_ae":
                print(f"Semantic reconstruction loss enabled: recon_weight={self.semantic_recon_weight}, "
                      f"l2_weight={semantic_l2_weight}, cosine_weight={semantic_cosine_weight}")
            if use_pixel_distill:
                print(f"Pixel distillation loss enabled with weight {self.pixel_distill_weight}")
        else:
            self.distill_loss = None
            print("Distillation loss disabled")

        # Note: Semantic KL loss is removed (using simplified PS-VAE without KL)
        self.semantic_kl_loss = None

        self.config = config
    
    def calculate_adaptive_weight(
        self, 
        base_loss: torch.Tensor, 
        aux_loss: torch.Tensor, 
        last_layer: torch.Tensor,
        base_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Calculate adaptive weight for auxiliary loss based on gradient norms.
        Reference: SVG (https://github.com/svg-project/SVG)
        
        The weight is computed as: base_weight * ||grad(base_loss)|| / ||grad(aux_loss)||
        This balances the auxiliary loss to have similar gradient magnitude as the base loss.
        
        Args:
            base_loss: The reference loss (e.g., reconstruction loss)
            aux_loss: The auxiliary loss to be weighted (e.g., vf_loss, distill_loss)
            last_layer: The layer to compute gradients w.r.t.
            base_weight: Base weight multiplier
            
        Returns:
            Adaptive weight for the auxiliary loss
        """
        try:
            base_grads = torch.autograd.grad(base_loss, last_layer, retain_graph=True)[0]
            aux_grads = torch.autograd.grad(aux_loss, last_layer, retain_graph=True)[0]
            
            adaptive_weight = torch.norm(base_grads) / (torch.norm(aux_grads) + 1e-4)
            adaptive_weight = torch.clamp(adaptive_weight, 0.0, self.adaptive_weight_max).detach()
            adaptive_weight = adaptive_weight * base_weight
            
            return adaptive_weight
        except RuntimeError:
            # If gradient computation fails (e.g., during eval), return base weight
            return torch.tensor(base_weight, device=base_loss.device)

    @autocast(enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                last_layer: torch.Tensor = None,
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Forward pass.
        
        Args:
            inputs: Input images
            reconstructions: Reconstructed images
            extra_result_dict: Dictionary containing extra outputs from encoder
            global_step: Current training step
            mode: "generator" or "discriminator"
            last_layer: Last layer of encoder for adaptive weight calculation
        """
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step, last_layer)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step: int):
        return global_step >= self.discriminator_iter_start

    def _normalize_inputs(self, inputs, reconstructions):
        """Normalize inputs and reconstructions to [0, 1] range."""
        # reverse ImageNet normalization
        std = torch.tensor([0.229, 0.224, 0.225]).to(inputs.device)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(inputs.device)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        
        # inputs is normalized by imagenet, while reconstruction predicts [-1, 1]
        # align both to [0, 1]
        inputs = inputs * std + mean
        reconstructions = (reconstructions + 1) / 2
        
        return inputs, reconstructions

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int,
                           last_layer: torch.Tensor = None,
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        
        # Normalize inputs
        inputs_norm, reconstructions_norm = self._normalize_inputs(inputs, reconstructions)

        # Compute reconstruction loss (used as base for adaptive weight)
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs_norm, reconstructions_norm, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs_norm, reconstructions_norm, reduction="mean")
        else:
            raise ValueError(f"Unsupported reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss_weighted = reconstruction_loss * self.reconstruction_weight

        # Compute perceptual loss
        perceptual_loss = self.perceptual_loss(inputs_norm, reconstructions_norm).mean()
        
        # Base loss for adaptive weight calculation (reconstruction + perceptual)
        base_loss_for_adaptive = reconstruction_loss_weighted + self.perceptual_weight * perceptual_loss

        # Compute distillation loss, semantic reconstruction loss, and pixel distill loss if enabled
        distill_loss = torch.tensor(0.0).to(inputs.device)
        layerwise_distill_loss = torch.tensor(0.0).to(inputs.device)
        semantic_recon_loss = torch.tensor(0.0).to(inputs.device)
        pixel_distill_loss = torch.tensor(0.0).to(inputs.device)
        semantic_loss_dict = {}
        
        if self.use_distill and 'distill_feat' in extra_result_dict:
            out_feat = extra_result_dict['distill_feat']
            
            # Get semantic_reconstructed if PS-VAE is enabled
            semantic_reconstructed = extra_result_dict.get('semantic_reconstructed', None)
            
            # Get pixel_latent if dual-stream is enabled
            pixel_latent = extra_result_dict.get('pixel_latent', None)
            
            # Get layer-wise features if layer-wise distillation is enabled
            student_layer_features = extra_result_dict.get('layer_features', None)
            
            # Compute distillation losses:
            # - distill_loss: semantic_feat (dual-stream) vs frozen encoder
            # - layerwise_distill_loss: ViT encoder layers vs frozen encoder layers (independent)
            distill_loss, layerwise_distill_loss, semantic_recon_loss, pixel_distill_loss, distill_loss_dict = self.distill_loss(
                inputs_norm, out_feat, semantic_reconstructed, pixel_latent, student_layer_features
            )
            semantic_loss_dict.update(distill_loss_dict)
        
        # Compute VF Loss for pixel stream if enabled
        vf_loss = torch.tensor(0.0).to(inputs.device)
        vf_loss_dict = {}
        if self.use_vf_loss and 'pixel_latent' in extra_result_dict:
            pixel_latent = extra_result_dict['pixel_latent']
            # Get DC-AE target from distill_loss module
            if self.distill_loss is not None and hasattr(self.distill_loss, 'dc_ae_encoder'):
                with torch.no_grad():
                    dc_ae_target = self.distill_loss.get_dc_ae_features(inputs_norm)
                vf_loss, vf_loss_dict = compute_vf_loss(
                    pixel_latent, dc_ae_target,
                    distmat_margin=self.vf_distmat_margin,
                    cos_margin=self.vf_cos_margin,
                    distmat_weight=self.vf_distmat_weight,
                    cos_weight=self.vf_cos_weight
                )

        # Compute discriminator loss
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions_norm)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight
        
        # Calculate adaptive weights if enabled
        adaptive_distill_w = self.distill_weight
        adaptive_pixel_distill_w = self.pixel_distill_weight
        adaptive_vf_w = self.vf_loss_weight
        
        if self.use_adaptive_weight and last_layer is not None and self.training:
            # Adaptive weight for semantic distillation loss
            if self.adaptive_distill_weight and self.use_distill and distill_loss.requires_grad:
                adaptive_distill_w = self.calculate_adaptive_weight(
                    base_loss_for_adaptive, distill_loss, last_layer, self.distill_weight
                )
            
            # Adaptive weight for pixel distillation loss
            if self.adaptive_pixel_distill_weight and (self.arch_type in ["dual_stream_trans", "dual_stream_simple", "dual_stream_pixeltrans"]) and self.pixel_distill_weight > 0.0:
                if pixel_distill_loss.requires_grad:
                    adaptive_pixel_distill_w = self.calculate_adaptive_weight(
                        base_loss_for_adaptive, pixel_distill_loss, last_layer, self.pixel_distill_weight
                    )
            
            # Adaptive weight for VF loss
            if self.adaptive_vf_weight and self.use_vf_loss and vf_loss.requires_grad:
                adaptive_vf_w = self.calculate_adaptive_weight(
                    base_loss_for_adaptive, vf_loss, last_layer, self.vf_loss_weight
                )

        # Build total loss based on quantize mode
        if self.quantize_mode == "vq":
            # VQ mode
            quantizer_loss = extra_result_dict.get("quantizer_loss", torch.tensor(0.0).to(inputs.device))
            total_loss = (
                reconstruction_loss_weighted
                + self.perceptual_weight * perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + adaptive_distill_w * distill_loss
                + d_weight * discriminator_factor * generator_loss
            )
            
            # Add layer-wise distillation loss if enabled (independent of distill_loss)
            # This loss aligns ViT encoder layers with frozen encoder layers
            if self.use_layerwise_distill and self.layerwise_distill_weight > 0.0:
                total_loss = total_loss + self.layerwise_distill_weight * layerwise_distill_loss
            
            # Add semantic AE loss if enabled (without KL)
            if self.arch_type == "semantic_ae":
                total_loss = total_loss + self.semantic_recon_weight * semantic_recon_loss
            
            # Add pixel distillation loss if dual-stream is enabled
            if self.arch_type in ["dual_stream_trans", "dual_stream_simple", "dual_stream_pixeltrans"] and self.pixel_distill_weight > 0.0:
                total_loss = total_loss + adaptive_pixel_distill_w * pixel_distill_loss
            
            # Add VF loss if enabled
            if self.use_vf_loss:
                total_loss = total_loss + adaptive_vf_w * vf_loss
            
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss_weighted.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor),
                commitment_loss=extra_result_dict.get("commitment_loss", torch.tensor(0.0)).detach(),
                codebook_loss=extra_result_dict.get("codebook_loss", torch.tensor(0.0)).detach(),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            if self.use_distill:
                loss_dict["distill_loss"] = (adaptive_distill_w * distill_loss).detach()
                # Record adaptive weight only if enabled
                if self.use_adaptive_weight and self.adaptive_distill_weight:
                    if isinstance(adaptive_distill_w, torch.Tensor):
                        loss_dict["adaptive_distill_weight"] = adaptive_distill_w.detach()
                    else:
                        loss_dict["adaptive_distill_weight"] = torch.tensor(adaptive_distill_w)
            
            # Add layer-wise distillation loss to dict if enabled
            if self.use_layerwise_distill and self.layerwise_distill_weight > 0.0:
                loss_dict["layerwise_distill_loss"] = (self.layerwise_distill_weight * layerwise_distill_loss).detach()
                # Add detailed layer-wise loss components from semantic_loss_dict
                for k, v in semantic_loss_dict.items():
                    if k.startswith('layerwise_'):
                        loss_dict[k] = v
            
            # Add semantic AE loss to dict if enabled (without KL)
            if self.arch_type == "semantic_ae":
                loss_dict["semantic_recon_loss"] = (self.semantic_recon_weight * semantic_recon_loss).detach()
                # Add detailed semantic loss components
                for k, v in semantic_loss_dict.items():
                    if not k.startswith('layerwise_'):
                        loss_dict[k] = v
            
            # Add pixel distillation loss to dict if dual-stream is enabled
            if self.arch_type in ["dual_stream_trans", "dual_stream_simple", "dual_stream_pixeltrans"] and self.pixel_distill_weight > 0.0:
                loss_dict["pixel_distill_loss"] = (adaptive_pixel_distill_w * pixel_distill_loss).detach()
                # Record adaptive weight only if enabled
                if self.use_adaptive_weight and self.adaptive_pixel_distill_weight:
                    if isinstance(adaptive_pixel_distill_w, torch.Tensor):
                        loss_dict["adaptive_pixel_distill_weight"] = adaptive_pixel_distill_w.detach()
                    else:
                        loss_dict["adaptive_pixel_distill_weight"] = torch.tensor(adaptive_pixel_distill_w)
            
            # Add VF loss to dict if enabled
            if self.use_vf_loss:
                loss_dict["vf_loss"] = (adaptive_vf_w * vf_loss).detach()
                # Record adaptive weight only if enabled
                if self.use_adaptive_weight and self.adaptive_vf_weight:
                    if isinstance(adaptive_vf_w, torch.Tensor):
                        loss_dict["adaptive_vf_weight"] = adaptive_vf_w.detach()
                    else:
                        loss_dict["adaptive_vf_weight"] = torch.tensor(adaptive_vf_w)
                # Add detailed VF loss components
                for k, v in vf_loss_dict.items():
                    loss_dict[k] = v
                
        elif self.quantize_mode == "vae":
            # VAE mode
            reconstruction_loss_vae = reconstruction_loss_weighted / torch.exp(self.logvar)
            if self.kl_weight > 0.0 and 'posteriors' in extra_result_dict:
                posteriors = extra_result_dict['posteriors']
                kl_loss = posteriors.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            else:
                kl_loss = torch.tensor(0.0).to(inputs.device)
            
            total_loss = (
                reconstruction_loss_vae
                + self.perceptual_weight * perceptual_loss
                + adaptive_distill_w * distill_loss
                + self.kl_weight * kl_loss
                + d_weight * discriminator_factor * generator_loss
            )
            
            # Add layer-wise distillation loss if enabled (independent of distill_loss)
            if self.use_layerwise_distill and self.layerwise_distill_weight > 0.0:
                total_loss = total_loss + self.layerwise_distill_weight * layerwise_distill_loss
            
            # Add semantic AE loss if enabled (without KL)
            if self.arch_type == "semantic_ae":
                total_loss = total_loss + self.semantic_recon_weight * semantic_recon_loss
            
            # Add pixel distillation loss if dual-stream is enabled
            if self.arch_type in ["dual_stream_trans", "dual_stream_simple", "dual_stream_pixeltrans"] and self.pixel_distill_weight > 0.0:
                total_loss = total_loss + adaptive_pixel_distill_w * pixel_distill_loss
            
            # Add VF loss if enabled
            if self.use_vf_loss:
                total_loss = total_loss + adaptive_vf_w * vf_loss
            
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss_vae.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                kl_loss=(self.kl_weight * kl_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            if self.use_distill:
                loss_dict["distill_loss"] = (adaptive_distill_w * distill_loss).detach()
                # Record adaptive weight only if enabled
                if self.use_adaptive_weight and self.adaptive_distill_weight:
                    if isinstance(adaptive_distill_w, torch.Tensor):
                        loss_dict["adaptive_distill_weight"] = adaptive_distill_w.detach()
                    else:
                        loss_dict["adaptive_distill_weight"] = torch.tensor(adaptive_distill_w)
            
            # Add layer-wise distillation loss to dict if enabled
            if self.use_layerwise_distill and self.layerwise_distill_weight > 0.0:
                loss_dict["layerwise_distill_loss"] = (self.layerwise_distill_weight * layerwise_distill_loss).detach()
                # Add detailed layer-wise loss components from semantic_loss_dict
                for k, v in semantic_loss_dict.items():
                    if k.startswith('layerwise_'):
                        loss_dict[k] = v
            
            # Add semantic AE loss to dict if enabled (without KL)
            if self.arch_type == "semantic_ae":
                loss_dict["semantic_recon_loss"] = (self.semantic_recon_weight * semantic_recon_loss).detach()
                # Add detailed semantic loss components
                for k, v in semantic_loss_dict.items():
                    if not k.startswith('layerwise_'):
                        loss_dict[k] = v
            
            # Add pixel distillation loss to dict if dual-stream is enabled
            if self.arch_type in ["dual_stream_trans", "dual_stream_simple", "dual_stream_pixeltrans"] and self.pixel_distill_weight > 0.0:
                loss_dict["pixel_distill_loss"] = (adaptive_pixel_distill_w * pixel_distill_loss).detach()
                # Record adaptive weight only if enabled
                if self.use_adaptive_weight and self.adaptive_pixel_distill_weight:
                    if isinstance(adaptive_pixel_distill_w, torch.Tensor):
                        loss_dict["adaptive_pixel_distill_weight"] = adaptive_pixel_distill_w.detach()
                    else:
                        loss_dict["adaptive_pixel_distill_weight"] = torch.tensor(adaptive_pixel_distill_w)
            
            # Add VF loss to dict if enabled
            if self.use_vf_loss:
                loss_dict["vf_loss"] = (adaptive_vf_w * vf_loss).detach()
                # Record adaptive weight only if enabled
                if self.use_adaptive_weight and self.adaptive_vf_weight:
                    if isinstance(adaptive_vf_w, torch.Tensor):
                        loss_dict["adaptive_vf_weight"] = adaptive_vf_w.detach()
                    else:
                        loss_dict["adaptive_vf_weight"] = torch.tensor(adaptive_vf_w)
                # Add detailed VF loss components
                for k, v in vf_loss_dict.items():
                    loss_dict[k] = v
        else:
            raise NotImplementedError(f"quantize_mode {self.quantize_mode} not supported")

        return total_loss, loss_dict

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discriminator training step."""
        # Normalize inputs
        inputs_norm, reconstructions_norm = self._normalize_inputs(inputs, reconstructions)
        
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        loss_dict = {}
        
        # Turn the gradients on
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs_norm.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions_norm.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # Optional LeCam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach() * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach() * (1 - self.lecam_ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict


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
"""
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

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
        
        # Semantic AE config (PS-VAE, without KL)
        # Based on paper: https://arxiv.org/pdf/2512.17909
        # Reference: vlvae_intervl_semae.py (simplified version without KL)
        self.use_semantic_ae = config.model.get("use_semantic_ae", False)
        self.semantic_recon_weight = loss_config.get("semantic_recon_weight", 1.0)
        
        # Dual stream config
        self.use_dual_stream = config.model.get("use_dual_stream", False)
        self.pixel_distill_weight = loss_config.get("pixel_distill_weight", 0.0)
        
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
            
            # Get DC-AE path for pixel distillation in dual-stream mode
            dc_ae_path = config.model.get("dc_ae_path", None) if self.use_dual_stream else None
            use_pixel_distill = self.use_dual_stream and self.pixel_distill_weight > 0.0
            
            self.distill_loss = DistillLoss(
                distill_loss_path,
                use_semantic_loss=self.use_semantic_ae,
                semantic_l2_weight=semantic_l2_weight,
                semantic_cosine_weight=semantic_cosine_weight,
                use_pixel_distill=use_pixel_distill,
                dc_ae_path=dc_ae_path
            ).eval()
            print(f"Distillation loss enabled with weight {self.distill_weight}")
            if self.use_semantic_ae:
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

    @autocast(enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Forward pass."""
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
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
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        
        # Normalize inputs
        inputs_norm, reconstructions_norm = self._normalize_inputs(inputs, reconstructions)

        # Compute reconstruction loss
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs_norm, reconstructions_norm, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs_norm, reconstructions_norm, reduction="mean")
        else:
            raise ValueError(f"Unsupported reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss
        perceptual_loss = self.perceptual_loss(inputs_norm, reconstructions_norm).mean()

        # Compute distillation loss, semantic reconstruction loss, and pixel distill loss if enabled
        distill_loss = torch.tensor(0.0).to(inputs.device)
        semantic_recon_loss = torch.tensor(0.0).to(inputs.device)
        pixel_distill_loss = torch.tensor(0.0).to(inputs.device)
        semantic_loss_dict = {}
        
        if self.use_distill and 'distill_feat' in extra_result_dict:
            out_feat = extra_result_dict['distill_feat']
            
            # Get semantic_reconstructed if PS-VAE is enabled
            semantic_reconstructed = extra_result_dict.get('semantic_reconstructed', None)
            
            # Get pixel_latent if dual-stream is enabled
            pixel_latent = extra_result_dict.get('pixel_latent', None)
            
            # Compute distillation loss (and semantic recon loss and pixel distill loss if enabled)
            distill_loss, semantic_recon_loss, pixel_distill_loss, distill_loss_dict = self.distill_loss(
                inputs_norm, out_feat, semantic_reconstructed, pixel_latent
            )
            semantic_loss_dict.update(distill_loss_dict)

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

        # Build total loss based on quantize mode
        if self.quantize_mode == "vq":
            # VQ mode
            quantizer_loss = extra_result_dict.get("quantizer_loss", torch.tensor(0.0).to(inputs.device))
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + self.distill_weight * distill_loss
                + d_weight * discriminator_factor * generator_loss
            )
            
            # Add semantic AE loss if enabled (without KL)
            if self.use_semantic_ae:
                total_loss = total_loss + self.semantic_recon_weight * semantic_recon_loss
            
            # Add pixel distillation loss if dual-stream is enabled
            if self.use_dual_stream and self.pixel_distill_weight > 0.0:
                total_loss = total_loss + self.pixel_distill_weight * pixel_distill_loss
            
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
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
                loss_dict["distill_loss"] = (self.distill_weight * distill_loss).detach()
            
            # Add semantic AE loss to dict if enabled (without KL)
            if self.use_semantic_ae:
                loss_dict["semantic_recon_loss"] = (self.semantic_recon_weight * semantic_recon_loss).detach()
                # Add detailed semantic loss components
                for k, v in semantic_loss_dict.items():
                    loss_dict[k] = v
            
            # Add pixel distillation loss to dict if dual-stream is enabled
            if self.use_dual_stream and self.pixel_distill_weight > 0.0:
                loss_dict["pixel_distill_loss"] = (self.pixel_distill_weight * pixel_distill_loss).detach()
                # Add raw pixel distill loss from distill_loss_dict
                for k, v in semantic_loss_dict.items():
                    if 'pixel' in k:
                        loss_dict[k] = v
                
        elif self.quantize_mode == "vae":
            # VAE mode
            reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
            if self.kl_weight > 0.0 and 'posteriors' in extra_result_dict:
                posteriors = extra_result_dict['posteriors']
                kl_loss = posteriors.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            else:
                kl_loss = torch.tensor(0.0).to(inputs.device)
            
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + self.distill_weight * distill_loss
                + self.kl_weight * kl_loss
                + d_weight * discriminator_factor * generator_loss
            )
            
            # Add semantic AE loss if enabled (without KL)
            if self.use_semantic_ae:
                total_loss = total_loss + self.semantic_recon_weight * semantic_recon_loss
            
            # Add pixel distillation loss if dual-stream is enabled
            if self.use_dual_stream and self.pixel_distill_weight > 0.0:
                total_loss = total_loss + self.pixel_distill_weight * pixel_distill_loss
            
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                kl_loss=(self.kl_weight * kl_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            if self.use_distill:
                loss_dict["distill_loss"] = (self.distill_weight * distill_loss).detach()
            
            # Add semantic AE loss to dict if enabled (without KL)
            if self.use_semantic_ae:
                loss_dict["semantic_recon_loss"] = (self.semantic_recon_weight * semantic_recon_loss).detach()
                # Add detailed semantic loss components
                for k, v in semantic_loss_dict.items():
                    loss_dict[k] = v
            
            # Add pixel distillation loss to dict if dual-stream is enabled
            if self.use_dual_stream and self.pixel_distill_weight > 0.0:
                loss_dict["pixel_distill_loss"] = (self.pixel_distill_weight * pixel_distill_loss).detach()
                # Add raw pixel distill loss from distill_loss_dict
                for k, v in semantic_loss_dict.items():
                    if 'pixel' in k:
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


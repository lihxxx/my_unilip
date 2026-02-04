from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sana import build_sana

from unilip.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_IDX, DEFAULT_IM_START_TOKEN_IDX, DEFAULT_IM_END_TOKEN_IDX, UND_IMAGE_TOKEN_IDX, DEFAULT_IMAGE_PATCH_TOKEN
import math
from transformers import AutoTokenizer, AutoModel, AutoConfig
from .vae_modules import DCAE_Decoder, ResBlock
from omegaconf import OmegaConf
from diffusers.models import AutoencoderDC
from copy import deepcopy


def build_alignment_mlp(hidden_size, projector_dim, z_dim):
    """构建Alignment Distill投影头MLP (与VGT/REPA一致)"""
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )

class UniLIP_InternVL_MetaModel:

    def __init__(self, config):
        super(UniLIP_InternVL_MetaModel, self).__init__(config)

        if hasattr(config, "n_query"):
            path = config.mllm_path
            internvl_model = AutoModel.from_pretrained(
                path,
                torch_dtype=self.vision_tower.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True)
            self.vision_tower = internvl_model.vision_model
            self.multi_modal_projector = internvl_model.mlp1

            for layer in self.vision_tower.encoder.layers:
                try:
                    layer.drop_path1.drop_prob = 0.0
                    layer.drop_path2.drop_prob = 0.0
                except:
                    continue
            print("should no drop out", self.vision_tower)
            self.vision_tower.eval()
            self.multi_modal_projector.eval()

            if 'hidden_size' in self.config:
                hidden_size = self.config.hidden_size
            else:
                hidden_size = self.config.text_config.hidden_size
            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, hidden_size))
            print(f" latent query size {self.latent_queries.shape}")

            self.dit, self.vae, self.noise_scheduler = build_sana(config.dit_path)

            # load unilip vae decoder
            vae_config = {
                'model':{
                    'dc_ae_path': config.vae_path
                }
            }
            vae_config = OmegaConf.create(vae_config)
            llm_hidden_size = self.multi_modal_projector[-1].weight.shape[-1]
            
            # Get dual stream config from config if available
            use_dual_stream = getattr(config, 'use_dual_stream', False)
            dual_stream_config = None
            if use_dual_stream:
                dual_stream_config = {
                    'num_layers': getattr(config, 'dual_stream_num_layers', 3),
                    'num_heads': getattr(config, 'dual_stream_num_heads', 16),
                    'mlp_ratio': getattr(config, 'dual_stream_mlp_ratio', 4.0),
                    'dropout': getattr(config, 'dual_stream_dropout', 0.0),
                    'use_cross_stream': getattr(config, 'use_cross_stream', False),
                    'cross_stream_num_heads': getattr(config, 'cross_stream_num_heads', 16),
                }
            self.vae_decoder = DCAE_Decoder(vae_config, llm_hidden_size, use_dual_stream, dual_stream_config)

            path = config.mllm_hf_path
            internvl_model = AutoModel.from_pretrained(
                path,
                torch_dtype=self.vision_tower.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager")
            self.llm_connector = deepcopy(internvl_model.language_model)
            del self.llm_connector.layers[:-self.config.connect_layer]
            del self.llm_connector.embed_tokens
            self.projector = nn.Linear(llm_hidden_size, self.dit.config.caption_channels)

    def initialize_vision_modules(self, model_args, fsdp=None):
        unilip_path = model_args.unilip_path
        self.unilip_path = unilip_path
        self.unilip_factor = model_args.unilip_factor
        self.fix_dit = model_args.fix_dit
        self.fix_connect = model_args.fix_connect
        print("fix connect", self.fix_connect)
        print("fix dit", self.fix_dit)
        if getattr(self, 'vae_decoder', None) is None:
            # replace hf structure with original internvl structure
            path = model_args.mllm_path
            internvl_model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True)
            self.vision_tower = internvl_model.vision_model
            self.multi_modal_projector = internvl_model.mlp1

            for layer in self.vision_tower.encoder.layers:
                try:
                    layer.drop_path1.drop_prob = 0.0
                    layer.drop_path2.drop_prob = 0.0
                except:
                    continue
            print("should no drop out", self.vision_tower)
            self.vision_tower.eval()

            # load unilip pretrain weight
            print(f"load from unilip path {unilip_path}, factor {self.unilip_factor}")
            unilip_ckpt = torch.load(unilip_path)
            encoder_state = {}
            for key, value in unilip_ckpt.items():
                if 'encoder.' in key:
                    newkey = key[len('encoder.'):]
                    encoder_state[newkey] = value

            msg = self.vision_tower.load_state_dict(encoder_state)
            for p in self.vision_tower.parameters():
                p.requires_grad = False
            print("load unilip vision encoder", msg)

            mlp1_state = {}
            for key, value in unilip_ckpt.items():
                if 'mlp1.' in key:
                    newkey = key[len('mlp1.'):]
                    mlp1_state[newkey] = value
            msg = self.multi_modal_projector.load_state_dict(mlp1_state)
            for p in self.multi_modal_projector.parameters():
                p.requires_grad = False
            print("load unilip mlp1", msg)

            # load vae decoder
            vae_config = {
                'model':{
                    'dc_ae_path': model_args.vae_path
                }
            }
            vae_config = OmegaConf.create(vae_config)
            llm_hidden_size = self.multi_modal_projector[-1].weight.shape[-1]
            
            # Get dual stream config from model_args if available
            use_dual_stream = getattr(model_args, 'use_dual_stream', False)
            dual_stream_config = None
            if use_dual_stream:
                dual_stream_config = {
                    'num_layers': getattr(model_args, 'dual_stream_num_layers', 3),
                    'num_heads': getattr(model_args, 'dual_stream_num_heads', 16),
                    'mlp_ratio': getattr(model_args, 'dual_stream_mlp_ratio', 4.0),
                    'dropout': getattr(model_args, 'dual_stream_dropout', 0.0),
                    'use_cross_stream': getattr(model_args, 'use_cross_stream', False),
                    'cross_stream_num_heads': getattr(model_args, 'cross_stream_num_heads', 16),
                }
            self.vae_decoder = DCAE_Decoder(vae_config, llm_hidden_size, use_dual_stream, dual_stream_config)
            
            # Filter unilip checkpoint keys for vae decoder
            decoder_ckpt = {}
            for name, value in unilip_ckpt.items():
                if 'regressor' in name:
                    continue
                # Include decoder and down-related keys
                if 'decoder' in name or 'down' in name:
                    decoder_ckpt[name] = value
                # For dual stream, also include semantic, pixel, fusion, and cross_stream related keys
                if use_dual_stream:
                    if any(key in name for key in ['semantic', 'pixel', 'fusion', 'cross_stream']):
                        decoder_ckpt[name] = value
            msg = self.vae_decoder.load_state_dict(decoder_ckpt, strict=False)
            for p in self.vae_decoder.parameters():
                p.requires_grad = False
            print("load unilip decoder", msg)
        else:
            print("unilip load from checkpoint!!!")
            self.vision_tower.eval()
            for p in self.vision_tower.parameters():
                p.requires_grad = False
            for p in self.multi_modal_projector.parameters():
                p.requires_grad = False
            for p in self.vae_decoder.parameters():
                p.requires_grad = False

        if getattr(self, 'dit', None) is None:
            print("random initiation the DiT !!!")
            self.dit, self.vae, self.noise_scheduler = build_sana(model_args.dit_path)
        else:
            print("DiT load from checkpoint!!!")
            for p in self.dit.parameters():
                p.requires_grad = True
        if self.fix_dit:
            for p in self.dit.parameters():
                p.requires_grad = False
        
        if getattr(self, 'llm_connector', None) is None:
            print("initialize the llm connector !!!")
            path = model_args.mllm_hf_path
            internvl_model = AutoModel.from_pretrained(
                path,
                torch_dtype=self.vision_tower.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager") # for bidr attention
            self.llm_connector = deepcopy(internvl_model.language_model)
            del self.llm_connector.layers[:-model_args.connect_layer]
            del self.llm_connector.embed_tokens
            self.projector = nn.Linear(llm_hidden_size, self.dit.config.caption_channels)
        else:
            print("Connector load from checkpoint!!!")
            for p in self.llm_connector.parameters():
                p.requires_grad = True
            for p in self.projector.parameters():
                p.requires_grad = True

        self.config.n_query = model_args.n_query
        self.config.connect_layer = model_args.connect_layer
        self.config.mllm_path = model_args.mllm_path
        self.config.mllm_hf_path = model_args.mllm_hf_path
        self.config.vae_path = model_args.vae_path
        self.config.dit_path = model_args.dit_path
        self.config.unilip_factor = model_args.unilip_factor

        if getattr(self, 'latent_queries', None) is None:
            print("random initiation the latent_queries !!!")
            if 'hidden_size' in self.config:
                hidden_size = self.config.hidden_size
            else:
                hidden_size = self.config.text_config.hidden_size
            self.latent_queries = nn.Parameter(torch.randn(1, self.config.n_query, hidden_size))
        else:
            print("latent_queries load from checkpoint!!!")
            self.latent_queries.requires_grad = True
        
        connect_require_grad = not self.fix_connect
        for p in self.llm_connector.parameters():
            p.requires_grad = connect_require_grad
        for p in self.projector.parameters():
            p.requires_grad = connect_require_grad
        self.latent_queries.requires_grad = connect_require_grad

        # ========== Unified Distill Loss 初始化 ==========
        # 统一的蒸馏损失框架，包含两个子损失：
        # 1. Alignment Loss: DiT中间层 -> projector -> 对齐 vit_proj_features
        # 2. Semantic Distill Loss: semantic_feat (经过cross stream) -> 对齐 vit_proj_features
        # 两者使用相同的目标特征：vit_proj_features（经过pixel_shuffle + mlp1）
        
        self.enable_alignment_loss = getattr(model_args, 'enable_repa', False)  # 兼容旧参数名
        self.alignment_loss_weight = getattr(model_args, 'repa_loss_weight', 0.5)
        self.alignment_encoder_depth = getattr(model_args, 'repa_encoder_depth', 6)
        alignment_projector_dim = getattr(model_args, 'repa_projector_dim', 2048)
        
        self.enable_semantic_distill = getattr(model_args, 'enable_semantic_distill', False)
        self.semantic_distill_weight = getattr(model_args, 'semantic_distill_weight', 0.1)
        
        # 存储hook捕获的特征
        self._dit_intermediate_features = None
        self._dit_hook_handle = None
        
        # 获取投影后的特征维度（经过mlp1后的维度）
        proj_hidden_size = self.multi_modal_projector[-1].weight.shape[-1]
        
        print(f"=== Initializing Unified Distill Loss ===")
        print(f"  Target feature: vit_proj_features (after pixel_shuffle + mlp1, dim={proj_hidden_size})")
        
        # Alignment Distill Loss 初始化
        if self.enable_alignment_loss:
            print(f"  [Alignment Distill Loss] Enabled")
            print(f"    - Weight: {self.alignment_loss_weight}")
            print(f"    - DiT layer depth: {self.alignment_encoder_depth}")
            
            # 获取DiT transformer blocks的内部隐藏维度
            dit_config = self.dit.config
            if hasattr(dit_config, 'num_attention_heads') and hasattr(dit_config, 'attention_head_dim'):
                dit_hidden_size = dit_config.num_attention_heads * dit_config.attention_head_dim
            else:
                dit_hidden_size = self.dit.transformer_blocks[0].attn1.to_q.in_features
            
            # 创建对齐投影头：DiT中间层特征 -> 投影特征空间
            self.alignment_projector = build_alignment_mlp(
                dit_hidden_size, 
                alignment_projector_dim, 
                proj_hidden_size
            )
            print(f"    - Projector: DiT({dit_hidden_size}) -> {alignment_projector_dim} -> Proj({proj_hidden_size})")
            
            for p in self.alignment_projector.parameters():
                p.requires_grad = True
            
            # 注册hook到DiT的指定层
            self._register_dit_hook()
        else:
            self.alignment_projector = None
            print(f"  [Alignment Distill Loss] Disabled")
        
        # Semantic Distill Loss 初始化
        if self.enable_semantic_distill:
            use_dual_stream = getattr(model_args, 'use_dual_stream', False)
            if not use_dual_stream:
                print(f"  [Semantic Distill Loss] Disabled (requires use_dual_stream=True)")
                self.enable_semantic_distill = False
            else:
                print(f"  [Semantic Distill Loss] Enabled")
                print(f"    - Weight: {self.semantic_distill_weight}")
                print(f"    - Aligns semantic_feat (after cross-stream) with vit_proj_features")
        else:
            print(f"  [Semantic Distill Loss] Disabled")
    
    def _register_dit_hook(self):
        """在DiT的指定transformer层注册forward hook来捕获中间特征"""
        if self._dit_hook_handle is not None:
            self._dit_hook_handle.remove()
        
        # Sana DiT的transformer blocks在 dit.transformer_blocks
        if hasattr(self.dit, 'transformer_blocks'):
            target_layer_idx = min(self.alignment_encoder_depth - 1, len(self.dit.transformer_blocks) - 1)
            target_layer = self.dit.transformer_blocks[target_layer_idx]
            
            def hook_fn(module, input, output):
                # output通常是 (hidden_states, ...) 或直接是hidden_states
                if isinstance(output, tuple):
                    self._dit_intermediate_features = output[0].clone()
                else:
                    self._dit_intermediate_features = output.clone()
            
            self._dit_hook_handle = target_layer.register_forward_hook(hook_fn)
            print(f"    - Registered hook at DiT transformer_blocks[{target_layer_idx}]")
        else:
            print(f"    - Warning: Could not find transformer_blocks in DiT, Alignment Loss disabled")
            self.enable_alignment_loss = False

    def _pixel_shuffle(self, x, scale_factor=0.5):
        """Pixel shuffle operation to match tokenizer implementation.
        
        Args:
            x: (N, W, H, C) tensor
            scale_factor: 0.5 means downsample spatial by 2x, upsample channels by 4x
        Returns:
            (N, H*scale, W*scale, C//(scale**2)) tensor
        """
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_vit_proj_features(self, pixel_values):
        """
        提取Vision Encoder的投影特征（经过pixel_shuffle + mlp1）
        
        作为Unified Distill Loss的共同目标特征：
        - Alignment Loss: DiT中间层 -> projector -> 对齐此特征
        - Semantic Distill Loss: semantic_feat (经过cross stream) -> 对齐此特征
        
        与tokenizer中distill_loss._get_ref_features保持一致
        
        Args:
            pixel_values: [B, C, H, W] 输入图像
        Returns:
            vit_proj_features: [B, N, D] vision encoder的投影特征
        """
        with torch.no_grad():
            # 获取patch embeddings
            vit_embeds = self.vision_tower.embeddings(pixel_values)
            
            # 通过所有encoder layers
            for encoder_layer in self.vision_tower.encoder.layers:
                vit_embeds = encoder_layer(vit_embeds)
            
            # 去掉CLS token，reshape，pixel_shuffle，mlp1
            vit_embeds = vit_embeds[:, 1:, :].contiguous().float()
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self._pixel_shuffle(vit_embeds)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            vit_proj_features = self.multi_modal_projector(vit_embeds)
        
        return vit_proj_features

    def compute_alignment_loss(self, dit_features, vit_proj_features):
        """
        计算Alignment Distill损失（负余弦相似度，与VGT/REPA实现一致）
        
        用于对齐DiT中间层特征与Vision Encoder投影特征
        Loss = -mean(cosine_similarity(dit_features, vit_proj_features))
        
        Args:
            dit_features: [B, N_dit, D] DiT中间层特征（经过alignment_projector投影后）
            vit_proj_features: [B, N_vit, D] Vision Encoder投影特征
        Returns:
            alignment_loss: 对齐损失值（范围约[-1, 0]，越小表示对齐越好）
        """
        # 对齐序列长度（通过adaptive pooling）
        N_dit = dit_features.shape[1]
        N_vit = vit_proj_features.shape[1]
        
        if N_dit != N_vit:
            # 使用adaptive average pooling对齐长度
            vit_proj_features = vit_proj_features.permute(0, 2, 1)  # [B, D, N_vit]
            vit_proj_features = F.adaptive_avg_pool1d(vit_proj_features, N_dit)
            vit_proj_features = vit_proj_features.permute(0, 2, 1)  # [B, N_dit, D]
        
        # L2 normalization
        dit_norm = F.normalize(dit_features, dim=-1)
        vit_norm = F.normalize(vit_proj_features, dim=-1)
        
        # 负余弦相似度
        cos_sim = (dit_norm * vit_norm).sum(dim=-1)  # [B, N]
        alignment_loss = -cos_sim.mean()
        
        return alignment_loss
    
    def get_dit_intermediate_features(self):
        """获取hook捕获的DiT中间层特征"""
        return self._dit_intermediate_features
    
    def clear_dit_intermediate_features(self):
        """清除存储的中间特征"""
        self._dit_intermediate_features = None

    # ========== Semantic Distill Loss ==========
    def get_semantic_features_for_distill(self, vit_embeds):
        """
        获取经过cross stream attention后的semantic_feat用于语义蒸馏损失
        
        Args:
            vit_embeds: (B, N, llm_hidden_size) 来自vision encoder + projector的特征
        Returns:
            semantic_feat: (B, N, llm_hidden_size) 经过cross stream后的语义特征
        """
        if not hasattr(self, 'vae_decoder') or self.vae_decoder is None:
            raise ValueError("vae_decoder is not initialized")
        
        if not self.vae_decoder.use_dual_stream:
            raise ValueError("Semantic distill loss requires use_dual_stream=True")
        
        return self.vae_decoder.get_semantic_features_with_cross_stream(vit_embeds)
    
    def compute_semantic_distill_loss(self, semantic_feat, vit_proj_features):
        """
        计算语义蒸馏损失：semantic_feat vs vision encoder投影特征
        
        与tokenizer训练时的distill_loss保持一致：
        - vit_proj_features: 经过pixel_shuffle + mlp1的特征
        - semantic_feat: 经过dual stream的semantic_transformer + cross_stream的特征
        
        Args:
            semantic_feat: (B, N, D) 经过cross stream后的语义特征
            vit_proj_features: (B, N, D) vision encoder的投影特征（经过pixel_shuffle + mlp1）
        Returns:
            semantic_distill_loss: MSE损失值
        """
        # 确保序列长度一致
        N_sem = semantic_feat.shape[1]
        N_vit = vit_proj_features.shape[1]
        
        if N_sem != N_vit:
            # 使用adaptive average pooling对齐长度
            vit_proj_features = vit_proj_features.permute(0, 2, 1)  # [B, D, N_vit]
            vit_proj_features = F.adaptive_avg_pool1d(vit_proj_features, N_sem)
            vit_proj_features = vit_proj_features.permute(0, 2, 1)  # [B, N_sem, D]
        
        # MSE损失
        semantic_distill_loss = F.mse_loss(semantic_feat, vit_proj_features)
        
        return semantic_distill_loss


class UniLIP_InternVL_MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_n_query(self):
        return self.get_model().config.n_query

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.get_model().noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.get_model().noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        gen_images, und_images, grid_thw, i_s_pos, image_sizes=None
    ):
        # Unilip: use same vision encoder for gen. and und.
        if (gen_images is None and und_images is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None

        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy
        with torch.no_grad():
            prompt_image_embeds = self.model.get_image_features(
                    pixel_values=gen_images,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_sizes=image_sizes,
                )
            # (B, HW, C) -> (B, C, H, W), assume H==W
            prompt_image_embeds = self.model.vae_decoder.clip_down(prompt_image_embeds)
        target_image_embeds = torch.clone(prompt_image_embeds).detach()
        latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
        H = latent_queries.shape[-1]
        latent_queries = latent_queries.contiguous().view(-1, H)

        if not und_images is None:
            und_image_embeds = self.model.get_image_features(
                pixel_values=und_images,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )
        else:
            und_image_embeds = None

        image_idx = (input_ids == IMAGE_TOKEN_IDX)
        und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
        output_indicator = labels != -100
        input_indicator = labels == -100
        text_embeds = self.get_model().language_model.embed_tokens(input_ids)
        gen_img_idx = torch.logical_and(output_indicator, image_idx)
       
        text_embeds = text_embeds.clone() 
        text_embeds[gen_img_idx] = latent_queries.to(text_embeds.dtype)
        und_img_idx = torch.logical_and(input_indicator, und_image_idx)

        if not und_images is None:
            text_embeds[und_img_idx] = und_image_embeds.to(text_embeds.device).flatten(0,1)

        labels[image_idx] = -100

        target_image_embeds = target_image_embeds.mul_(self.model.unilip_factor)

        bidr_attention_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
        bidr_attention_mask = bidr_attention_mask.unsqueeze(1)
        bidr_attention_mask = (1-bidr_attention_mask.float())*-100000
        return None, position_ids, attention_mask, past_key_values, text_embeds, labels, target_image_embeds, und_img_idx, und_image_embeds, bidr_attention_mask



    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

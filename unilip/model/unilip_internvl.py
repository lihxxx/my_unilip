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


def build_repa_mlp(hidden_size, projector_dim, z_dim):
    """构建REPA投影头MLP"""
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
            self.vae_decoder = DCAE_Decoder(vae_config, llm_hidden_size)

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
            self.vae_decoder = DCAE_Decoder(vae_config, llm_hidden_size)
            for name in list(unilip_ckpt.keys()):
                if 'regressor' in name:
                    del unilip_ckpt[name]
                else:
                    if 'decoder' in name or 'down' in name:
                        continue
                    else:
                        del unilip_ckpt[name]
            msg = self.vae_decoder.load_state_dict(unilip_ckpt)
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

        # ========== REPA相关初始化 ==========
        # REPA: 对齐DiT中间层特征与Vision Encoder特征
        self.enable_repa = getattr(model_args, 'enable_repa', False)
        self.repa_loss_weight = getattr(model_args, 'repa_loss_weight', 0.5)
        self.repa_encoder_depth = getattr(model_args, 'repa_encoder_depth', 6)
        repa_projector_dim = getattr(model_args, 'repa_projector_dim', 2048)
        
        # 存储hook捕获的特征
        self._dit_intermediate_features = None
        self._dit_hook_handle = None
        
        if self.enable_repa:
            print(f"=== Initializing REPA ===")
            print(f"  REPA loss weight: {self.repa_loss_weight}")
            print(f"  REPA encoder depth (DiT layer): {self.repa_encoder_depth}")
            
            # 获取vision encoder的特征维度（作为REPA target）
            vit_hidden_size = self.vision_tower.embeddings.patch_embedding.weight.shape[0]
            
            # 获取DiT transformer blocks的内部隐藏维度
            # Sana DiT: inner_dim = num_attention_heads * attention_head_dim
            dit_config = self.dit.config
            if hasattr(dit_config, 'num_attention_heads') and hasattr(dit_config, 'attention_head_dim'):
                dit_hidden_size = dit_config.num_attention_heads * dit_config.attention_head_dim
            else:
                # fallback: 尝试从第一个transformer block获取
                dit_hidden_size = self.dit.transformer_blocks[0].attn1.to_q.in_features
            
            # 创建REPA投影头：将DiT中间层特征映射到vision encoder特征空间
            self.repa_projector = build_repa_mlp(
                dit_hidden_size, 
                repa_projector_dim, 
                vit_hidden_size  # 映射到vision encoder的特征维度
            )
            print(f"  REPA projector: DiT({dit_hidden_size}) -> {repa_projector_dim} -> ViT({vit_hidden_size})")
            
            # REPA投影头需要训练
            for p in self.repa_projector.parameters():
                p.requires_grad = True
            
            # 注册hook到DiT的指定层
            self._register_dit_hook()
        else:
            self.repa_projector = None
    
    def _register_dit_hook(self):
        """在DiT的指定transformer层注册forward hook来捕获中间特征"""
        if self._dit_hook_handle is not None:
            self._dit_hook_handle.remove()
        
        # Sana DiT的transformer blocks在 dit.transformer_blocks
        if hasattr(self.dit, 'transformer_blocks'):
            target_layer_idx = min(self.repa_encoder_depth - 1, len(self.dit.transformer_blocks) - 1)
            target_layer = self.dit.transformer_blocks[target_layer_idx]
            
            def hook_fn(module, input, output):
                # output通常是 (hidden_states, ...) 或直接是hidden_states
                if isinstance(output, tuple):
                    self._dit_intermediate_features = output[0].clone()
                else:
                    self._dit_intermediate_features = output.clone()
            
            self._dit_hook_handle = target_layer.register_forward_hook(hook_fn)
            print(f"  Registered REPA hook at DiT transformer_blocks[{target_layer_idx}]")
        else:
            print(f"  Warning: Could not find transformer_blocks in DiT, REPA disabled")
            self.enable_repa = False

    def extract_vision_features_for_repa(self, pixel_values):
        """
        提取vision encoder的中间层特征用于REPA对齐
        
        Args:
            pixel_values: [B, C, H, W] 输入图像
        Returns:
            vit_features: [B, N, D] vision encoder的patch特征
        """
        # 是否解冻vision encoder决定是否使用no_grad
        if not any(p.requires_grad for p in self.vision_tower.parameters()):
            # vision encoder冻结，使用no_grad
            with torch.no_grad():
                vit_features = self._extract_vit_features_impl(pixel_values)
        else:
            # REPA-E模式，vision encoder可训练
            vit_features = self._extract_vit_features_impl(pixel_values)
        
        return vit_features
    
    def _extract_vit_features_impl(self, pixel_values):
        """实际提取vision encoder特征的实现"""
        # 获取vision encoder的patch embeddings
        vit_embeds = self.vision_tower.embeddings(pixel_values)
        
        # 通过encoder layers（只到repa_encoder_depth）
        for idx, encoder_layer in enumerate(self.vision_tower.encoder.layers):
            vit_embeds = encoder_layer(vit_embeds)
            if idx + 1 >= self.repa_encoder_depth:
                break
        
        # 去掉CLS token，只保留patch tokens
        vit_features = vit_embeds[:, 1:, :].contiguous()
        
        return vit_features

    def compute_repa_loss(self, dit_features, vit_features):
        """
        计算REPA对齐损失（负余弦相似度）
        
        Args:
            dit_features: [B, N_dit, D] DiT中间层特征（经过projector投影后）
            vit_features: [B, N_vit, D] Vision Encoder中间层特征
        Returns:
            repa_loss: REPA损失值
        """
        # 对齐序列长度（通过adaptive pooling）
        N_dit = dit_features.shape[1]
        N_vit = vit_features.shape[1]
        
        if N_dit != N_vit:
            # 使用adaptive average pooling对齐长度
            # 将vit_features pooling到dit_features的长度
            vit_features = vit_features.permute(0, 2, 1)  # [B, D, N_vit]
            vit_features = F.adaptive_avg_pool1d(vit_features, N_dit)
            vit_features = vit_features.permute(0, 2, 1)  # [B, N_dit, D]
        
        # L2 normalization
        dit_norm = F.normalize(dit_features, dim=-1)
        vit_norm = F.normalize(vit_features, dim=-1)
        
        # 负余弦相似度
        cos_sim = (dit_norm * vit_norm).sum(dim=-1)  # [B, N]
        repa_loss = -cos_sim.mean()
        
        return repa_loss
    
    def get_dit_intermediate_features(self):
        """获取hook捕获的DiT中间层特征"""
        return self._dit_intermediate_features
    
    def clear_dit_intermediate_features(self):
        """清除存储的中间特征"""
        self._dit_intermediate_features = None


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

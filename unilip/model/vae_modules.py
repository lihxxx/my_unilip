# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torch.cuda.amp import autocast
from diffusers.models import AutoencoderDC

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
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


class CrossStreamAttention(nn.Module):
    """Cross-Stream Attention for Dual Stream Architecture.
    
    Enables information exchange between semantic and pixel streams.
    - Semantic stream can attend to pixel stream features
    - Pixel stream can attend to semantic stream features
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


class DiagonalGaussianDistribution(object):
    @autocast(enabled=False)
    def __init__(self, parameters, deterministic=False):
        """Initializes a Gaussian distribution instance given the parameters.

        Args:
            parameters (torch.Tensor): The parameters for the Gaussian distribution. It is expected
                to be in shape [B, 2 * C, *], where B is batch size, and C is the embedding dimension.
                First C channels are used for mean and last C are used for logvar in the Gaussian distribution.
            deterministic (bool): Whether to use deterministic sampling. When it is true, the sampling results
                is purely based on mean (i.e., std = 0).
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    @autocast(enabled=False)
    def sample(self):
        x = self.mean.float() + self.std.float() * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    @autocast(enabled=False)
    def mode(self):
        return self.mean

    @autocast(enabled=False)
    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean.float(), 2)
                                    + self.var.float() - 1.0 - self.logvar.float(),
                                    dim=[1, 2])

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h




class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError

class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    # def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
    #              attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
    #              resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
    #              **ignore_kwargs):
    def __init__(self, config):
        ch = config.model.vq_model.ch 
        out_ch = config.model.vq_model.out_ch 
        ch_mult = config.model.vq_model.ch_mult 
        num_res_blocks = config.model.vq_model.num_res_blocks 
        attn_resolutions = config.model.vq_model.attn_resolutions 
        dropout = config.model.vq_model.dropout 
        resamp_with_conv = True 
        in_channels = config.model.vq_model.in_channels 
        resolution = config.model.vq_model.resolution 
        z_channels = config.model.vq_model.z_channels 
        double_z = config.model.vq_model.double_z 
        use_linear_attn = False 
        attn_type = 'vanilla'

        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class DCAE_Decoder(nn.Module):
    """
    DCAE Decoder with optional Dual Stream support and Cross-Stream Attention.
    
    Dual Stream Architecture (when use_dual_stream=True):
    - Stream 1 (Semantic): vit_embeds -> semantic_transformer -> semantic_feat 
                          -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
    - Stream 2 (Pixel): vit_embeds -> pixel_transformer -> pixel_feat
                       -> pixel_down_blocks -> pixel_down_mlp -> 32-dim
    - Cross-Stream (optional): CrossStreamAttention for information exchange
    - Fusion: concat(32, 32) -> 64-dim -> fusion_layer -> 32-dim -> decoder
    
    Original Architecture (when use_dual_stream=False):
    - vit_embeds -> down_blocks -> down_mlp -> 32-dim -> decoder
    
    Args:
        config: Config containing model.dc_ae_path and optional dual_stream settings
        llm_hidden_size: Hidden size from LLM (e.g., 2048 for InternVL3-1B)
        use_dual_stream: Whether to use dual stream architecture (default: False)
        dual_stream_config: Dict with dual stream configuration:
            - num_layers: Number of transformer encoder layers (default: 3)
            - num_heads: Number of attention heads (default: 16)
            - mlp_ratio: MLP expansion ratio (default: 4.0)
            - dropout: Dropout rate (default: 0.0)
            - use_cross_stream: Whether to use cross-stream attention (default: False)
            - cross_stream_num_heads: Number of attention heads for cross-stream (default: 16)
    """
    
    def __init__(self, config, llm_hidden_size, use_dual_stream=False, dual_stream_config=None):
        super().__init__()
        
        self.use_dual_stream = use_dual_stream
        self.llm_hidden_size = llm_hidden_size
        
        # Load DC-AE decoder
        dc_ae = AutoencoderDC.from_pretrained(config.model.dc_ae_path, torch_dtype=torch.float32)
        self.decoder = dc_ae.decoder
        
        # Default dual stream config
        if dual_stream_config is None:
            dual_stream_config = {}
        self.dual_stream_num_layers = dual_stream_config.get('num_layers', 3)
        self.dual_stream_num_heads = dual_stream_config.get('num_heads', 16)
        self.dual_stream_mlp_ratio = dual_stream_config.get('mlp_ratio', 4.0)
        self.dual_stream_dropout = dual_stream_config.get('dropout', 0.0)
        # Cross-stream interaction config
        self.use_cross_stream = dual_stream_config.get('use_cross_stream', False)
        self.cross_stream_num_heads = dual_stream_config.get('cross_stream_num_heads', 16)
        
        if self.use_dual_stream:
            # ============ Dual Stream Architecture ============
            # Stream 1 (Semantic): vit_embeds -> semantic_transformer -> semantic_feat
            # Stream 2 (Pixel): vit_embeds -> pixel_transformer -> pixel_feat
            # Cross-Stream (optional): CrossStreamAttention for information exchange
            # Then: semantic_feat -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
            #       pixel_feat -> down_blocks -> down_mlp -> 32-dim
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
                print(f"  DCAE_Decoder: Cross-Stream Attention enabled with num_heads={self.cross_stream_num_heads}")
            
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
            
            print(f"DCAE_Decoder: Dual Stream enabled with num_layers={self.dual_stream_num_layers}, "
                  f"num_heads={self.dual_stream_num_heads}, use_cross_stream={self.use_cross_stream}")
        else:
            # ============ Original Single Stream Architecture ============
            # vit_embeds -> down_blocks -> down_mlp -> 32-dim
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
            print("DCAE_Decoder: Using original single stream architecture")

    def forward(self, vit_embeds):
        """
        Forward pass through the decoder.
        
        Args:
            vit_embeds: Input features of shape (B, N, llm_hidden_size)
            
        Returns:
            h: Reconstructed image of shape (B, 3, H, W)
        """
        if self.use_dual_stream:
            # Dual stream forward
            z = self._dual_stream_encode(vit_embeds)
        else:
            # Original single stream forward
            z = self._single_stream_encode(vit_embeds)
        
        h = self.decoder(z)
        return h
    
    def _single_stream_encode(self, vit_embeds):
        """Single stream encoding: vit_embeds -> down_blocks -> down_mlp -> latent"""
        for block in self.down_blocks:
            vit_embeds = block(vit_embeds)
        vit_embeds = self.down_mlp(vit_embeds)
        
        vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()
        
        b, c, hw = vit_embeds.shape
        z = vit_embeds.view(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))
        return z
    
    def _dual_stream_encode(self, vit_embeds):
        """
        Dual stream encoding with optional cross-stream attention:
        - Semantic: vit_embeds -> semantic_transformer -> semantic_feat
        - Pixel: vit_embeds -> pixel_transformer -> pixel_feat
        - Cross-Stream (optional): CrossStreamAttention for information exchange
        - Then: semantic_feat -> semantic_down_blocks -> semantic_down_mlp -> 32-dim
        -       pixel_feat -> pixel_down_blocks -> pixel_down_mlp -> 32-dim
        - Fusion: concat(semantic, pixel) -> fusion_layer -> 32-dim
        """
        # Semantic stream: vit_embeds -> TransformerEncoder -> semantic_feat
        semantic_feat = self.semantic_transformer(vit_embeds)  # (B, N, llm_hidden_size)
        
        # Pixel stream: vit_embeds -> TransformerEncoder -> pixel_feat
        pixel_feat = self.pixel_transformer(vit_embeds)  # (B, N, llm_hidden_size)
        
        # Cross-stream attention (optional): information exchange between streams
        if self.use_cross_stream:
            semantic_feat, pixel_feat = self.cross_stream_attention(semantic_feat, pixel_feat)
        
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
        
        # Fusion: concat(semantic_latent, pixel_latent) -> fusion_layer -> 32-dim
        fused_latent = torch.cat([semantic_latent, pixel_latent], dim=-1)  # (B, N, 64)
        latent_for_decoder = self.fusion_layer(fused_latent)  # (B, N, 32)
        
        # Reshape to spatial format
        latent_for_decoder = latent_for_decoder.permute(0, 2, 1).contiguous()
        
        b, c, hw = latent_for_decoder.shape
        z = latent_for_decoder.view(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))
        return z
    
    def clip_down(self, vit_embeds):
        """
        Encode vit_embeds to latent space without decoding.
        
        Args:
            vit_embeds: Input features of shape (B, N, llm_hidden_size)
            
        Returns:
            z: Latent representation of shape (B, 32, H, W)
        """
        if self.use_dual_stream:
            z = self._dual_stream_encode(vit_embeds)
        else:
            z = self._single_stream_encode(vit_embeds)
        return z
    
    def vae_decode(self, z):
        """
        Decode latent representation to image.
        
        Args:
            z: Latent representation of shape (B, 32, H, W)
            
        Returns:
            h: Reconstructed image of shape (B, 3, H, W)
        """
        h = self.decoder(z)
        return h
    
    def get_semantic_features(self, vit_embeds):
        """
        Get semantic features from the semantic stream (only available when use_dual_stream=True).
        
        This can be used for distillation loss computation.
        
        Args:
            vit_embeds: Input features of shape (B, N, llm_hidden_size)
            
        Returns:
            semantic_feat: Semantic features of shape (B, N, llm_hidden_size)
        """
        if not self.use_dual_stream:
            raise ValueError("get_semantic_features is only available when use_dual_stream=True")
        
        semantic_feat = self.semantic_transformer(vit_embeds)
        return semantic_feat
    
    def get_pixel_latent(self, vit_embeds):
        """
        Get pixel latent from the pixel stream (only available when use_dual_stream=True).
        
        This can be used for pixel distillation loss computation.
        
        Args:
            vit_embeds: Input features of shape (B, N, llm_hidden_size)
            
        Returns:
            pixel_latent: Pixel latent of shape (B, N, 32)
        """
        if not self.use_dual_stream:
            raise ValueError("get_pixel_latent is only available when use_dual_stream=True")
        
        pixel_latent = vit_embeds
        for block in self.pixel_down_blocks:
            pixel_latent = block(pixel_latent)
        pixel_latent = self.pixel_down_mlp(pixel_latent)
        return pixel_latent

class Decoder(nn.Module):
    # def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
    #              attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
    #              resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
    #              attn_type="vanilla", **ignorekwargs):
    def __init__(self, config):
        ch = config.model.vq_model.ch 
        out_ch = config.model.vq_model.out_ch 
        ch_mult = config.model.vq_model.ch_mult 
        num_res_blocks = config.model.vq_model.num_res_blocks 
        attn_resolutions = config.model.vq_model.attn_resolutions 
        dropout = config.model.vq_model.dropout 
        resamp_with_conv = True 
        in_channels = config.model.vq_model.in_channels 
        resolution = config.model.vq_model.resolution 
        z_channels = config.model.vq_model.z_channels 
        give_pre_end = False
        tanh_out = False 
        use_linear_attn = False 
        attn_type = 'vanilla'

        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class UpBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        block = nn.ModuleList()
        attn = nn.ModuleList()
        block_in = channels
        block_out = channels
        attn_type = 'vanilla'
        for i_block in range(3):
            block.append(ResnetBlock(in_channels=block_in,
                                        out_channels=block_out,
                                        temb_channels=0,
                                        dropout=0.0))
            block_in = block_out
            attn.append(make_attn(block_in, attn_type=attn_type))
        up = nn.Module()
        up.block = block
        up.attn = attn
        up.upsample = Upsample(block_in, True)
        self.up = up
    
    def forward(self, h):
        for i_block in range(3):
            h = self.up.block[i_block](h, None)
            if len(self.up.attn) > 0:
                h = self.up.attn[i_block](h)
        h = self.up.upsample(h)
        return h

import torch
from torch import nn
from torch.nn import functional as F
from functools import partial, reduce
import numpy as np
import typing as tp

import x_transformers
from x_transformers import ContinuousTransformerWrapper, Encoder
from einops import rearrange

from .blocks import ResConvBlock, FourierFeatures, Upsample1d, Upsample1d_2, Downsample1d, Downsample1d_2, SelfAttention1d, SkipBlock, expand_to_planes
from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..inference.generation import generate_diffusion_cond

from .adp import UNetCFG1d, UNet1d

from time import time

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep

class DiffusionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, t, **kwargs):
        raise NotImplementedError()

class DiffusionModelWrapper(nn.Module):
    def __init__(
                self,
                model: DiffusionModel,
                io_channels,
                sample_size,
                sample_rate,
                min_input_length,
                pretransform: Pretransform = None,
    ):
        super().__init__()
        self.io_channels = io_channels
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.min_input_length = min_input_length

        self.model = model

        if pretransform is not None:
            self.pretransform = pretransform
        else:
            self.pretransform = None

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)
    
class ConditionedDiffusionModel(nn.Module):
    def __init__(self, 
                *args,
                supports_cross_attention: bool = False,
                supports_input_concat: bool = False,
                supports_global_cond: bool = False,
                supports_prepend_cond: bool = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond
    
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor,   
                cross_attn_cond: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None,
                input_concat_cond: torch.Tensor = None,
                global_embed: torch.Tensor = None,
                prepend_cond: torch.Tensor = None,
                prepend_cond_mask: torch.Tensor = None,
                cfg_scale: float = 1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                **kwargs):
        raise NotImplementedError()
    
class ConditionedDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self, 
            model: ConditionedDiffusionModel,
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            pretransform: Pretransform = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            ):
        super().__init__()

        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.min_input_length = min_input_length

    def get_conditioning_inputs(self, cond: tp.Dict[str, tp.Any], negative=False):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None

        if len(self.cross_attn_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_input = torch.cat([cond[key][0] for key in self.cross_attn_cond_ids], dim=1)
            cross_attention_masks = torch.cat([cond[key][1] for key in self.cross_attn_cond_ids], dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_cond = torch.cat([cond[key][0] for key in self.global_cond_ids], dim=-1)
            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)
        
        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_cond = torch.cat([cond[key][0] for key in self.input_concat_ids], dim=1)

        if len(self.prepend_cond_ids) > 0:
            # Concatenate all prepend conditioning inputs over the sequence dimension
            # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
            prepend_cond = torch.cat([cond[key][0] for key in self.prepend_cond_ids], dim=1)
            prepend_cond_mask = torch.cat([cond[key][1] for key in self.prepend_cond_ids], dim=1)

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_global_cond": global_cond,
                "negative_input_concat_cond": input_concat_cond
            }
        else:
            return {
                "cross_attn_cond": cross_attention_input,
                "cross_attn_mask": cross_attention_masks,
                "global_cond": global_cond,
                "input_concat_cond": input_concat_cond,
                "prepend_cond": prepend_cond,
                "prepend_cond_mask": prepend_cond_mask
            }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)
    
    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)

class UNetCFG1DWrapper(ConditionedDiffusionModel):
    def __init__(
        self, 
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=True, supports_input_concat=True)

        self.model = UNetCFG1d(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self, 
                x, 
                t, 
                cross_attn_cond=None, 
                cross_attn_mask=None, 
                input_concat_cond=None, 
                global_cond=None, 
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                negative_global_cond=None,
                negative_input_concat_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                **kwargs):
        p = Profiler()

        p.tick("start")

        channels_list = None
        if input_concat_cond is not None:
            channels_list = [input_concat_cond]

        outputs = self.model(
            x, 
            t, 
            embedding=cross_attn_cond, 
            embedding_mask=cross_attn_mask, 
            features=global_cond, 
            channels_list=channels_list,
            embedding_scale=cfg_scale, 
            embedding_mask_proba=cfg_dropout_prob, 
            batch_cfg=batch_cfg,
            rescale_cfg=rescale_cfg,
            negative_embedding=negative_cross_attn_cond,
            negative_embedding_mask=negative_cross_attn_mask,
            **kwargs)
        
        p.tick("UNetCFG1D forward")

        #print(f"Profiler: {p}")
        return outputs

class UNet1DCondWrapper(ConditionedDiffusionModel):
    def __init__(
        self, 
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=False, supports_global_cond=True, supports_input_concat=True)

        self.model = UNet1d(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self, 
                x, 
                t, 
                input_concat_cond=None, 
                global_cond=None, 
                **kwargs):

        channels_list = None
        if input_concat_cond is not None:
            channels_list = [input_concat_cond]

        outputs = self.model(
            x, 
            t, 
            features=global_cond, 
            channels_list=channels_list,
            **kwargs)
    
        return outputs

class UNet1DUncondWrapper(DiffusionModel):
    def __init__(
        self, 
        in_channels,
        *args,        
        **kwargs
    ):
        super().__init__()

        self.model = UNet1d(in_channels=in_channels, *args, **kwargs)

        self.io_channels = in_channels

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs) 

class DAU1DCondWrapper(ConditionedDiffusionModel):
    def __init__(
        self, 
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=False, supports_global_cond=False, supports_input_concat=True)

        self.model = DiffusionAttnUnet1D(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self, 
                x, 
                t, 
                input_concat_cond=None, 
                cross_attn_cond=None, 
                cross_attn_mask=None, 
                global_cond=None, 
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                negative_global_cond=None,
                negative_input_concat_cond=None,
                prepend_cond=None,
                **kwargs):
        
        return self.model(x, t, cond = input_concat_cond)

class DiffusionAttnUnet1D(nn.Module):
    def __init__(
        self, 
        io_channels = 2, 
        depth=14,
        n_attn_layers = 6,
        channels = [128, 128, 256, 256] + [512] * 10,
        cond_dim = 0,
        cond_noise_aug = False,
        kernel_size = 5,
        learned_resample = False,
        strides = [2] * 13,
        conv_bias = True,
        use_snake = False
    ):
        super().__init__()

        self.cond_noise_aug = cond_noise_aug

        self.io_channels = io_channels

        if self.cond_noise_aug:
            self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_embed = FourierFeatures(1, 16)

        attn_layer = depth - n_attn_layers

        strides = [1] + strides

        block = nn.Identity()

        conv_block = partial(ResConvBlock, kernel_size=kernel_size, conv_bias = conv_bias, use_snake=use_snake)

        for i in range(depth, 0, -1):
            c = channels[i - 1]
            stride = strides[i-1]
            if stride > 2 and not learned_resample:
                raise ValueError("Must have stride 2 without learned resampling")
            
            if i > 1:
                c_prev = channels[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d_2(c_prev, c_prev, stride) if (learned_resample or stride == 1) else Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if add_attn else nn.Identity(),
                    Upsample1d_2(c_prev, c_prev, stride) if learned_resample else Upsample1d(kernel="cubic")
                )
            else:
                cond_embed_dim = 16 if not self.cond_noise_aug else 32
                block = nn.Sequential(
                    conv_block((io_channels + cond_dim) + cond_embed_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, io_channels, is_last=True),
                )
        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, x, t, cond=None, cond_aug_scale=None):

        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)
        
        inputs = [x, timestep_embed]

        if cond is not None:
            if cond.shape[2] != x.shape[2]:
                cond = F.interpolate(cond, (x.shape[2], ), mode='linear', align_corners=False)
                
            if self.cond_noise_aug:
                # Get a random number between 0 and 1, uniformly sampled
                if cond_aug_scale is None:
                    aug_level = self.rng.draw(cond.shape[0])[:, 0].to(cond)  
                else:
                    aug_level = torch.tensor([cond_aug_scale]).repeat([cond.shape[0]]).to(cond)             

                # Add noise to the conditioning signal
                cond = cond + torch.randn_like(cond) * aug_level[:, None, None]

                # Get embedding for noise cond level, reusing timestamp_embed
                aug_level_embed = expand_to_planes(self.timestep_embed(aug_level[:, None]), x.shape)

                inputs.append(aug_level_embed)

            inputs.append(cond)

        outputs = self.net(torch.cat(inputs, dim=1))

        return outputs
        
class DiTWrapper(ConditionedDiffusionModel):
    def __init__(
        self, 
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False)

        self.model = DiffusionTransformer(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self, 
                x, 
                t, 
                cross_attn_cond=None, 
                cross_attn_mask=None, 
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                input_concat_cond=None, 
                negative_input_concat_cond=None,
                global_cond=None, 
                negative_global_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = True,
                rescale_cfg: bool = False,
                scale_phi: float = 0.0,
                **kwargs):

        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
        assert negative_input_concat_cond is None, "negative_input_concat_cond is not supported for DiTWrapper"
        assert global_cond is None, "global_cond is not supported for DiTWrapper"
        assert negative_global_cond is None, "negative_global_cond is not supported for DiTWrapper"

        return self.model(
            x, 
            t, 
            cross_attn_cond=cross_attn_cond, 
            cross_attn_cond_mask=cross_attn_mask, 
            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale=cfg_scale, 
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            **kwargs)    

class DiTUncondWrapper(DiffusionModel):
    def __init__(
        self, 
        in_channels,
        *args,        
        **kwargs
    ):
        super().__init__()

        self.model = DiffusionTransformer(io_channels=in_channels, *args, **kwargs)

        self.io_channels = in_channels

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs) 

class DiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        input_length=512,
        embed_dim=768,
        cond_token_dim=0,
        global_cond_dim=0,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        **kwargs):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        if cond_token_dim > 0:
            # Conditioning tokens
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if global_cond_dim > 0:
            # Global conditioning
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        # RoPE monkey patch fix

        def rotate_half(x):
            x = rearrange(x, '... (j d) -> ... j d', j = 2)
            x1, x2 = x.unbind(dim = -2)
            return torch.cat((-x2, x1), dim = -1)

        def apply_rotary_pos_emb(t, freqs, scale = 1):
            out_dtype = t.dtype
            dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))

            rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
            freqs = freqs[-seq_len:, :]

            if t.ndim == 4 and freqs.ndim == 3:
                freqs = rearrange(freqs, 'b n d -> b 1 n d')

            t, freqs = t.to(dtype), freqs.to(dtype)

            # partial rotary embeddings, Wang et al. GPT-J
            t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
            t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

            t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

            return torch.cat((t, t_unrotated), dim = -1) 

        x_transformers.x_transformers.apply_rotary_pos_emb = torch.compile(apply_rotary_pos_emb)

        # Transformer

        self.transformer = ContinuousTransformerWrapper(
            dim_in=dim_in,
            dim_out=io_channels,
            max_seq_len=0, #Not relevant without absolute positional embeds
            attn_layers = Encoder(
                dim=embed_dim,
                depth=depth,
                heads=num_heads,
                attn_flash = True,
                cross_attend = cond_token_dim > 0,
                zero_init_branch_output=True,
                use_abs_pos_emb = False,
                rotary_pos_emb=True,
                ff_swish = True,
                ff_glu = True,
                **kwargs
            )
        )

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def _forward(
        self, 
        x, 
        t, 
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        **kwargs):

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        # Add a sequence dimension to the timestep embeddings
        timestep_embed = timestep_embed.unsqueeze(1)

        prepend_inputs = timestep_embed

        prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

            # Add the global conditioning to the timestep embeddings
            prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(2)], dim=2)           

            prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)

            # Set up inputs to prepend to transformer inputs
            prepend_inputs = torch.cat([prepend_inputs, prepend_cond], dim=1)
            
            if prepend_cond_mask is not None:
                prepend_mask = torch.cat([prepend_mask, prepend_cond_mask], dim=1)
            else:
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], prepend_cond.shape[1]), device=x.device, dtype=torch.bool)], dim=1)

        prepend_length = prepend_inputs.shape[1]

        if input_concat_cond is not None:
            x = torch.cat([x, input_concat_cond], dim=1)

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, **kwargs)

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        output = self.postprocess_conv(output) + output

        return output

    def forward(
        self, 
        x, 
        t, 
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        causal=False,
        scale_phi=0.0,
        mask=None,
        **kwargs):

        assert causal == False, "Causal mode is not supported for DiffusionTransformer"

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        if cross_attn_cond is not None:

            # CFG dropout
            if cfg_dropout_prob > 0.0:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

        if cross_attn_cond is not None and cfg_scale != 1.0:
            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            
            null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if prepend_cond is not None:
                batch_prepend_cond = torch.cat([prepend_cond, prepend_cond], dim=0)
            else:
                batch_prepend_cond = None

            if prepend_cond_mask is not None:
                batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
            else:
                batch_prepend_cond_mask = None

            if negative_cross_attn_cond is not None:
                if negative_cross_attn_mask is not None:
                    negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                    negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                
                batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

            else:
                batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

            if cross_attn_cond_mask is not None:
                batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
            else:
                batch_cond_masks = None

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None
            
            batch_output = self._forward(
                batch_inputs, 
                batch_timestep, 
                cross_attn_cond=batch_cond, 
                cross_attn_cond_mask=batch_cond_masks, 
                mask = batch_masks, 
                input_concat_cond=input_concat_cond, 
                global_embed = batch_global_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                **kwargs)

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            if scale_phi != 0.0:

                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)

                return scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output

            else:

                return cfg_output
            
        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond, 
                cross_attn_cond_mask=cross_attn_cond_mask, 
                input_concat_cond=input_concat_cond, 
                global_embed=global_embed, 
                prepend_cond=prepend_cond, 
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                **kwargs
            )

def create_diffusion_uncond_from_config(config: tp.Dict[str, tp.Any]):
    diffusion_uncond_config = config["model"]

    model_type = diffusion_uncond_config.get('type', None)

    diffusion_config = diffusion_uncond_config.get('config', {})

    assert model_type is not None, "Must specify model type in config"

    pretransform = diffusion_uncond_config.get("pretransform", None)

    sample_size = config.get("sample_size", None)
    assert sample_size is not None, "Must specify sample size in config"

    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "Must specify sample rate in config"

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    if model_type == 'DAU1d':

        model = DiffusionAttnUnet1D(
            **diffusion_config
        )

        
    
    elif model_type == "adp_uncond_1d":

        model = UNet1DUncondWrapper(
            **diffusion_config
        )
    
    elif model_type == "dit":
        model = DiTUncondWrapper(
            **diffusion_config
        )

    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

    return DiffusionModelWrapper(model, 
                                io_channels=model.io_channels, 
                                sample_size=sample_size, 
                                sample_rate=sample_rate,
                                pretransform=pretransform,
                                min_input_length=min_input_length)
    
def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):

    model_config = config["model"]

    diffusion_config = model_config.get('diffusion', None)
    assert diffusion_config is not None, "Must specify diffusion config"

    diffusion_model_type = diffusion_config.get('type', None)
    assert diffusion_model_type is not None, "Must specify diffusion model type"
    
    diffusion_model_config = diffusion_config.get('config', None)
    assert diffusion_model_config is not None, "Must specify diffusion model config"

    if diffusion_model_type == 'adp_cfg_1d':
        diffusion_model = UNetCFG1DWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'dit':
        diffusion_model = DiTWrapper(**diffusion_model_config)

    io_channels = model_config.get('io_channels', None)
    assert io_channels is not None, "Must specify io_channels in model config"

    sample_rate = config.get('sample_rate', None)
    assert sample_rate is not None, "Must specify sample_rate in config"

    conditioning_config = model_config.get('conditioning', None)

    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)

    cross_attention_ids = diffusion_config.get('cross_attention_cond_ids', [])
    global_cond_ids = diffusion_config.get('global_cond_ids', [])
    input_concat_ids = diffusion_config.get('input_concat_ids', [])
    prepend_cond_ids = diffusion_config.get('prepend_cond_ids', [])

    pretransform = model_config.get("pretransform", None)
    
    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    if diffusion_model_type == "adp_cfg_1d":
        min_input_length *= np.prod(diffusion_model_config["factors"])
    elif diffusion_model_type == "dit":
        min_input_length = min_input_length # There's no downsampling in DiT

    return ConditionedDiffusionModelWrapper(
        diffusion_model, 
        conditioner, 
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=pretransform, 
        io_channels=io_channels
    )
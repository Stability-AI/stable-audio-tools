import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
import typing as tp

from x_transformers import ContinuousTransformerWrapper, Encoder
from einops import rearrange

from .blocks import ResConvBlock, FourierFeatures, Upsample1d, Upsample1d_2, Downsample1d, Downsample1d_2, SelfAttention1d, SkipBlock, expand_to_planes
from .factory import create_pretransform_from_config
from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
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
                pretransform: Pretransform = None
    ):
        super().__init__()
        self.io_channels = io_channels
        self.sample_size = sample_size
        self.sample_rate = sample_rate

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
                **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
    
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor,   
                cross_attn_cond: torch.Tensor = None,
                cross_attn_masks: torch.Tensor = None,
                input_concat_cond: torch.Tensor = None,
                global_embed: torch.Tensor = None,
                cfg_scale: float = 1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                scale_cfg: bool = False,
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
            pretransform: Pretransform = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = []
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

    def get_conditioning_inputs(self, cond: tp.Dict[str, tp.Any]):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None

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

        return {
            "cross_attn_cond": cross_attention_input,
            "cross_attn_masks": cross_attention_masks,
            "global_cond": global_cond,
            "input_concat_cond": input_concat_cond
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
                cross_attn_masks=None, 
                input_concat_cond=None, 
                global_cond=None, 
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                scale_cfg: bool = False,
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
            embedding_mask=cross_attn_masks, 
            features=global_cond, 
            channels_list=channels_list,
            embedding_scale=cfg_scale, 
            embedding_mask_proba=cfg_dropout_prob, 
            batch_cfg=batch_cfg,
            scale_cfg=scale_cfg,
            **kwargs)
        
        p.tick("UNetCFG1D forward")

        #print(f"Profiler: {p}")
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
                cross_attn_masks=None, 
                input_concat_cond=None, 
                global_cond=None, 
                cfg_scale=6.0,
                cfg_dropout_prob: float = 0.1,
                batch_cfg: bool = True,
                scale_cfg: bool = False,
                **kwargs):

        assert batch_cfg, "batch_cfg must be True for DiTWrapper"

        return self.model(
            x, 
            t, 
            cond=cross_attn_cond, 
            cond_mask=cross_attn_masks, 
            cfg_scale=cfg_scale, 
            cfg_dropout_prob=cfg_dropout_prob,
            **kwargs)    

class DiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        input_length=512,
        cond_token_dim=0,
        embed_dim=768,
        depth=12,
        num_heads=8):

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

        # Transformer

        self.transformer = ContinuousTransformerWrapper(
            dim_in=io_channels,
            dim_out=io_channels,
            max_seq_len=input_length + 1, #1 for time conditioning
            attn_layers = Encoder(
                dim=embed_dim,
                depth=depth,
                heads=num_heads,
                attn_flash = True,
                cross_attend = True,
                zero_init_branch_output=True,
                rotary_pos_emb =True,
                ff_swish = True, # set this to True
                ff_glu = True 
            )
        )

        self.preprocess_conv = nn.Conv1d(io_channels, io_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def forward(
        self, 
        x, 
        t, 
        cond=None,
        cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0):

        if cond_mask is not None:
            cond_mask = cond_mask.bool()

            cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        timestep_embed = timestep_embed.unsqueeze(1)

        if cond is not None:

            cond = self.to_cond_embed(cond)

            # CFG dropout
            if cfg_dropout_prob > 0.0:
                null_embed = torch.zeros_like(cond, device=cond.device)
                dropout_mask = torch.bernoulli(torch.full((cond.shape[0], 1, 1), cfg_dropout_prob, device=cond.device)).to(torch.bool)
                cond = torch.where(dropout_mask, null_embed, cond)

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        if cond is not None and cfg_scale != 1.0:
            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            
            null_embed = torch.zeros_like(cond, device=cond.device)

            batch_timestep = torch.cat([timestep_embed, timestep_embed], dim=0)
            batch_cond = torch.cat([cond, null_embed], dim=0)
            if cond_mask is not None:
                batch_masks = torch.cat([cond_mask, cond_mask], dim=0)
            else:
                batch_masks = None
            
            output = self.transformer(batch_inputs, prepend_embeds=batch_timestep, context=batch_cond, context_mask=batch_masks)

            cond_output, uncond_output = torch.chunk(output, 2, dim=0)
            output = uncond_output + (cond_output - uncond_output) * cfg_scale
            
        else:
            output = self.transformer(x, prepend_embeds=timestep_embed, context=cond, context_mask=cond_mask)

        output = rearrange(output, "b t c -> b c t")[:,:,1:]

        output = self.postprocess_conv(output) + output

        return output

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

    if model_type == 'DAU1d':

        model = DiffusionAttnUnet1D(
            **diffusion_config
        )

        return DiffusionModelWrapper(model, 
                                     io_channels=model.io_channels, 
                                     sample_size=sample_size, 
                                     sample_rate=sample_rate,
                                     pretransform=pretransform)
    
    elif model_type == "adp_uncond_1d":

        model = UNet1DUncondWrapper(
            **diffusion_config
        )

        return DiffusionModelWrapper(model, 
                                     io_channels=model.io_channels, 
                                     sample_size=sample_size, 
                                     sample_rate=sample_rate,
                                     pretransform=pretransform)
    
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
    
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
    assert conditioning_config is not None, "Must specify conditioning config"

    conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)

    cross_attention_ids = diffusion_config.get('cross_attention_cond_ids', [])
    global_cond_ids = diffusion_config.get('global_cond_ids', [])
    input_concat_ids = diffusion_config.get('input_concat_ids', [])

    pretransform = model_config.get("pretransform", None)
    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    return ConditionedDiffusionModelWrapper(
        diffusion_model, 
        conditioner, 
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        pretransform=pretransform, 
        io_channels=io_channels
    )
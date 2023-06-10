import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from typing import Dict, Any

from .blocks import ResConvBlock, FourierFeatures, Upsample1d, Upsample1d_2, Downsample1d, Downsample1d_2, SelfAttention1d, SkipBlock, expand_to_planes
from .factory import create_pretransform_from_config

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
        conv_bias = True
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

        conv_block = partial(ResConvBlock, kernel_size=kernel_size, conv_bias = conv_bias)

        for i in range(depth, 0, -1):
            c = channels[i - 1]
            stride = strides[i-1]
            if stride != 2 and not learned_resample:
                raise ValueError("Must have stride 2 without learned resampling")
            
            if i > 1:
                c_prev = channels[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d_2(c_prev, c_prev, stride) if learned_resample else Downsample1d("cubic"),
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
  
class DiffusionModel(nn.Module):
    def __init__(
                self,
                model,
                io_channels = 2,
                pretransform = None
    ):
        super().__init__()
        self.model = model
        self.io_channels = io_channels

        if pretransform is not None:
            self.pretransform = pretransform

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

def create_diffusion_from_config(model_config: Dict[str, Any]):
    model_type = model_config.get('type', None)

    diffusion_config = model_config.get('config', {})

    assert model_type is not None, "Must specify model type in config"

    pretransform = model_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform)

    if model_type == 'DAU1d':
        model = DiffusionAttnUnet1D(
            **diffusion_config
        )

        return DiffusionModel(model, io_channels=model.io_channels, pretransform=pretransform)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
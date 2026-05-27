import numpy as np
import torch
import torch.nn.functional as F
import typing as tp

from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .utils import load_ckpt_state_dict, copy_state_dict

from .blocks import FourierFeatures

def get_relativistic_losses(score_real, score_fake):
    # Compute difference between real and fake scores
    diff = score_real - score_fake
    dis_loss = F.softplus(-diff).mean()
    gen_loss = F.softplus(diff).mean()
    return dis_loss, gen_loss

class ConvDiscriminator(nn.Module):
    def __init__(self, channels, hidden_dim=None, soft_clip_scale=None, loss_type="lsgan", anti_aliased=False):
        super().__init__()

        self.loss_type = loss_type
        self.anti_aliased = anti_aliased 
        
        hidden_dim = hidden_dim if hidden_dim is not None else channels

        self.layers = nn.Sequential(
            nn.Conv1d(kernel_size=4, in_channels=channels, out_channels=hidden_dim, stride=2, padding=1), 
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv1d(kernel_size=4, in_channels=hidden_dim, out_channels=hidden_dim, stride=2, padding=1), 
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv1d(kernel_size=4, in_channels=hidden_dim, out_channels=hidden_dim, stride=2, padding=1), 
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv1d(kernel_size=4, in_channels=hidden_dim, out_channels=hidden_dim, stride=2, padding=1), 
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv1d(kernel_size=4, in_channels=hidden_dim, out_channels=1, stride=1, padding=0), 
        )

        self.soft_clip_scale = soft_clip_scale

    def reset_parameters(self):
        with torch.no_grad():
            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, x):
        # x shape: [Batch, Channels, Time]
        
        if self.anti_aliased:
            # Pass 1: Normal View
            out_normal = self.layers(x)
            
            # Pass 2: Phase-Shifted View
            # Roll input by 1 on time axis so strides see the "odd" pixels
            x_shifted = torch.roll(x, shifts=1, dims=-1)
            out_shifted = self.layers(x_shifted)
            
            # Concatenate along Batch Dimension (dim 0)
            # Output becomes [2*Batch, 1, Time_Out]
            output = torch.cat([out_normal, out_shifted], dim=0)
            
        else:
            output = self.layers(x)

        if self.soft_clip_scale is not None:
            output = self.soft_clip_scale * torch.tanh(output/self.soft_clip_scale)

        return output
        

class ConvNeXtDiscriminator(nn.Module):
    def __init__(self, loss_type="lsgan", *args, **kwargs):
        super().__init__()

        from .convnext import ConvNeXtEncoder

        self.encoder = ConvNeXtEncoder(*args, **kwargs)

        self.loss_type = loss_type

    def reset_parameters(self):
        with torch.no_grad():
            self.encoder.reset_parameters()

    def forward(self, x):
        return self.encoder(x)

    def loss(self, reals, fakes, *args, **kwargs):
        real_scores = self(reals)
        fake_scores = self(fakes)

        loss_dis = loss_adv = 0

        if self.loss_type == "lsgan":
            # Calculate least-squares GAN losses
            loss_dis = torch.mean(fake_scores**2) + torch.mean ((1 - real_scores)**2)
            loss_adv = torch.mean((1 - fake_scores)**2)
        elif self.loss_type == "relativistic":
            
            diff = real_scores - fake_scores

            loss_dis = F.softplus(-diff).mean()
            loss_adv = F.softplus(diff).mean()

        return {
            "loss_dis": loss_dis,
            "loss_adv": loss_adv
        }

class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.convs = nn.Sequential(
            weight_norm(nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation)),
            nn.GroupNorm(num_groups=16, num_channels=channels), # GroupNorm is stable for GANs
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(channels, channels, kernel_size=3, dilation=1, padding=1)),
            nn.GroupNorm(num_groups=16, num_channels=channels),
            nn.LeakyReLU(0.2)
        )
        # Skip connection is identity since channels match
        
    def forward(self, x):
        return x + self.convs(x)

class DilatedConvDiscriminator(nn.Module):
    def __init__(self, channels, hidden_dim=64, loss_type="lsgan", dilations=[1, 2, 4, 1, 2, 4]):
        super().__init__()
        self.loss_type = loss_type

        # 1. Input Projection
        self.start_conv = weight_norm(nn.Conv1d(channels, hidden_dim, kernel_size=3, padding=1))

        # 2. Dilated Stack (configurable)
        self.layers = nn.ModuleList([
            ResidualDilatedBlock(hidden_dim, dilation=d) for d in dilations
        ])
        
        # 3. Final Scoring
        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)) # Stride 1 output
        )

    def reset_parameters(self):
        with torch.no_grad():
            # Reset start_conv (weight_norm wraps the weight)
            if hasattr(self.start_conv, 'reset_parameters'):
                self.start_conv.reset_parameters()
            # Reset dilated blocks
            for layer in self.layers:
                for sublayer in layer.convs:
                    if hasattr(sublayer, 'reset_parameters'):
                        sublayer.reset_parameters()
            # Reset final_conv
            for sublayer in self.final_conv:
                if hasattr(sublayer, 'reset_parameters'):
                    sublayer.reset_parameters()

    def forward(self, x):
        h = self.start_conv(x)
        for layer in self.layers:
            h = layer(h)
        return self.final_conv(h)

    def loss(self, reals, fakes, **kwargs):
        real_scores = self(reals)
        fake_scores = self(fakes)
        
        loss_dis = loss_adv = 0

        if self.loss_type == "lsgan":
             loss_dis = torch.mean(fake_scores**2) + torch.mean((1 - real_scores)**2)
             loss_adv = torch.mean((1 - fake_scores)**2)
        elif self.loss_type == "relativistic":
             diff = real_scores - fake_scores
             loss_dis = F.softplus(-diff).mean()
             loss_adv = F.softplus(diff).mean()
             
        return {"loss_dis": loss_dis, "loss_adv": loss_adv}


class TransformerDiscriminator(nn.Module):
    """
    Sliding window differential attention discriminator head.

    Uses transformer blocks with sliding window attention - a "convolutional hypernetwork"
    that combines local receptive fields with input-dependent, learned weights.

    Advantages over dilated convolutions:
    - Dense coverage (no dilation holes)
    - Input-dependent weights (adapts to content)
    - Clean boundary handling (no padding artifacts)
    - Position-aware via RoPE
    """
    def __init__(
        self,
        channels,
        hidden_dim=256,
        depth=4,
        dim_heads=64,
        sliding_window=[2, 2],
        differential=True,
        qk_norm='rms',
        ff_mult=2,
        loss_type="lsgan",
    ):
        super().__init__()

        from .transformer import TransformerBlock, RMSNorm

        self.loss_type = loss_type
        self.sliding_window = sliding_window

        # Input projection
        self.proj_in = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            RMSNorm(hidden_dim)
        )

        # Transformer blocks with sliding window attention
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                dim_heads=dim_heads,
                causal=False,
                zero_init_branch_outputs=True,
                add_rope=True,
                norm_type='rms_norm',
                attn_kwargs={
                    'qk_norm': qk_norm,
                    'differential': differential
                },
                ff_kwargs={'mult': ff_mult}
            )
            for _ in range(depth)
        ])

        # Output projection to scores
        self.to_scores = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def reset_parameters(self):
        """Recursively reset all parameters in the discriminator head."""
        with torch.no_grad():
            for module in self.modules():
                if module is not self and hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        x = self.proj_in(x)

        for layer in self.layers:
            x = layer(x, self_attention_flash_sliding_window=self.sliding_window)

        scores = self.to_scores(x)  # [B, T, 1]
        return scores.transpose(1, 2)  # [B, 1, T]

    def loss(self, reals, fakes, **kwargs):
        real_scores = self(reals)
        fake_scores = self(fakes)

        loss_dis = loss_adv = 0

        if self.loss_type == "lsgan":
            loss_dis = torch.mean(fake_scores**2) + torch.mean((1 - real_scores)**2)
            loss_adv = torch.mean((1 - fake_scores)**2)
        elif self.loss_type == "relativistic":
            diff = real_scores - fake_scores
            loss_dis = F.softplus(-diff).mean()
            loss_adv = F.softplus(diff).mean()

        return {"loss_dis": loss_dis, "loss_adv": loss_adv}


class MultiScaleTransformerDiscriminator(nn.Module):
    """
    Multi-scale discriminator head using TransformerResamplingBlock sub-heads.

    Each sub-head operates at a different temporal stride, targeting different
    aspects of audio quality:
    - Local (stride=1): texture, transients, sharpness
    - Mid (stride=4): phrase-level coherence
    - Global (stride=16): overall structure, arrangement, energy

    Returns a list of score tensors, one per head, each [B, 1, T_out].
    """
    def __init__(
        self,
        channels,
        heads=None,
        hidden_dim=256,
        depth=3,
        dim_heads=64,
        sliding_window=None,
        differential=True,
        ff_mult=2,
        dyt=False,
        checkpointing=True,
        loss_type="lsgan",
    ):
        super().__init__()

        from .autoencoders import TransformerResamplingBlock
        from .transformer import RMSNorm

        self.loss_type = loss_type

        if heads is None:
            heads = [
                {"name": "local",  "stride": 1,  "hidden_dim": 256, "depth": 2, "sliding_window": [3, 3]},
                {"name": "mid",    "stride": 4,  "hidden_dim": 256, "depth": 3, "sliding_window": [12, 12]},
                {"name": "global", "stride": 16, "hidden_dim": 256, "depth": 3, "sliding_window": [96, 96]},
            ]

        self.head_names = [h.get("name", f"scale_{i}") for i, h in enumerate(heads)]
        self.head_strides = [h.get("stride", 1) for h in heads]
        self.sub_heads = nn.ModuleList()
        self.score_projs = nn.ModuleList()

        for head_config in heads:
            h_hidden = head_config.get("hidden_dim", hidden_dim)
            h_depth = head_config.get("depth", depth)
            h_stride = head_config.get("stride", 1)
            h_window = head_config.get("sliding_window", sliding_window)
            h_dim_heads = head_config.get("dim_heads", dim_heads)
            h_differential = head_config.get("differential", differential)
            h_ff_mult = head_config.get("ff_mult", ff_mult)
            h_dyt = head_config.get("dyt", dyt)

            block = TransformerResamplingBlock(
                in_channels=channels,
                out_channels=h_hidden,
                stride=h_stride,
                sliding_window=h_window,
                type='encoder',
                transformer_depth=h_depth,
                dim_heads=h_dim_heads,
                differential=h_differential,
                use_flash=True,
                checkpointing=checkpointing,
                dyt=h_dyt,
                ff_mult=h_ff_mult,
                mask_noise=0,
            )
            self.sub_heads.append(block)

            self.score_projs.append(nn.Sequential(
                RMSNorm(h_hidden),
                nn.Linear(h_hidden, 1),
            ))

    def reset_parameters(self):
        with torch.no_grad():
            for module in self.modules():
                if module is not self and hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

    def forward(self, x):
        """
        Args:
            x: [B, C, T] concatenated DiT hidden states

        Returns:
            List of [B, 1, T_i] score tensors, one per sub-head
        """
        scores_list = []
        for sub_head, score_proj in zip(self.sub_heads, self.score_projs):
            h = sub_head(x)            # [B, hidden_dim, T_out]
            h = h.transpose(1, 2)      # [B, T_out, hidden_dim]
            s = score_proj(h)           # [B, T_out, 1]
            scores_list.append(s.transpose(1, 2))  # [B, 1, T_out]
        return scores_list
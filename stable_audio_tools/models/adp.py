# Copied and modified from https://github.com/archinetai/audio-diffusion-pytorch/blob/v0.0.94/audio_diffusion_pytorch/modules.py under MIT License
# License can be found in LICENSES/LICENSE_ADP.txt

import math
from inspect import isfunction
from math import ceil, floor, log, pi, log2
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from packaging import version

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
from torch import Tensor, einsum
from torch.backends.cuda import sdp_kernel
from torch.nn import functional as F

"""
Utils
"""


class ConditionedSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(*modules)

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None):
        for module in self.module_list:
            x = module(x, mapping)
        return x

T = TypeVar("T")

def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val: Optional[T]) -> T:
    return val is not None

def closest_power_2(x: float) -> int:
    exponent = log2(x)
    distance_fn = lambda z: abs(x - 2 ** z)  # noqa
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)

def group_dict_by_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]:
    return_dicts: Tuple[Dict, Dict] = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts

def groupby(prefix: str, d: Dict, keep_prefix: bool = False) -> Tuple[Dict, Dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs

"""
Convolutional Blocks
"""
import typing as tp

# Copied from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conv.py under MIT License
# License available in LICENSES/LICENSE_META.txt

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: Tensor, causal=False) -> Tensor:
        kernel_size = self.kernel_size[0]
        stride = self.stride[0]
        dilation = self.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding))
        return super().forward(x)
        
class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, causal=False) -> Tensor:
        kernel_size = self.kernel_size[0]
        stride = self.stride[0]
        padding_total = kernel_size - stride

        y = super().forward(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if causal:
            padding_right = ceil(padding_total)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y
    

def Downsample1d(
    in_channels: int, out_channels: int, factor: int, kernel_multiplier: int = 2
) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor * kernel_multiplier + 1,
        stride=factor
    )


def Upsample1d(
    in_channels: int, out_channels: int, factor: int, use_nearest: bool = False
) -> nn.Module:

    if factor == 1:
        return Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3
        )

    if use_nearest:
        return nn.Sequential(
            nn.Upsample(scale_factor=factor, mode="nearest"),
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3
            ),
        )
    else:
        return ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * 2,
            stride=factor
        )


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
        use_norm: bool = True
    ) -> None:
        super().__init__()

        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            if use_norm
            else nn.Identity()
        )

        
        self.activation = nn.SiLU() 

        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(
        self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None, causal=False
    ) -> Tensor:
        x = self.groupnorm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.activation(x)
        return self.project(x, causal=causal)


class MappingToScaleShift(nn.Module):
    def __init__(
        self,
        features: int,
        channels: int,
    ):
        super().__init__()

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=features, out_features=channels * 2),
        )

    def forward(self, mapping: Tensor) -> Tuple[Tensor, Tensor]:
        scale_shift = self.to_scale_shift(mapping)
        scale_shift = rearrange(scale_shift, "b c -> b c 1")
        scale, shift = scale_shift.chunk(2, dim=1)
        return scale, shift


class ResnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        use_norm: bool = True,
        num_groups: int = 8,
        context_mapping_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_mapping = exists(context_mapping_features)

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            use_norm=use_norm,
            num_groups=num_groups
        )

        if self.use_mapping:
            assert exists(context_mapping_features)
            self.to_scale_shift = MappingToScaleShift(
                features=context_mapping_features, channels=out_channels
            )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            use_norm=use_norm,
            num_groups=num_groups
        )

        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None, causal=False) -> Tensor:
        assert_message = "context mapping required if context_mapping_features > 0"
        assert not (self.use_mapping ^ exists(mapping)), assert_message

        h = self.block1(x, causal=causal)

        scale_shift = None
        if self.use_mapping:
            scale_shift = self.to_scale_shift(mapping)

        h = self.block2(h, scale_shift=scale_shift, causal=causal)

        return h + self.to_out(x)


class Patcher(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        context_mapping_features: Optional[int] = None,
    ):
        super().__init__()
        assert_message = f"out_channels must be divisible by patch_size ({patch_size})"
        assert out_channels % patch_size == 0, assert_message
        self.patch_size = patch_size

        self.block = ResnetBlock1d(
            in_channels=in_channels,
            out_channels=out_channels // patch_size,
            num_groups=1,
            context_mapping_features=context_mapping_features
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None, causal=False) -> Tensor:
        x = self.block(x, mapping, causal=causal)
        x = rearrange(x, "b c (l p) -> b (c p) l", p=self.patch_size)
        return x


class Unpatcher(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        context_mapping_features: Optional[int] = None
    ):
        super().__init__()
        assert_message = f"in_channels must be divisible by patch_size ({patch_size})"
        assert in_channels % patch_size == 0, assert_message
        self.patch_size = patch_size

        self.block = ResnetBlock1d(
            in_channels=in_channels // patch_size,
            out_channels=out_channels,
            num_groups=1,
            context_mapping_features=context_mapping_features
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None, causal=False) -> Tensor:
        x = rearrange(x, " b (c p) l -> b c (l p) ", p=self.patch_size)
        x = self.block(x, mapping, causal=causal)
        return x


"""
Attention Components
"""
def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )

def add_mask(sim: Tensor, mask: Tensor) -> Tensor:
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim

def causal_mask(q: Tensor, k: Tensor) -> Tensor:
    b, i, j, device = q.shape[0], q.shape[-2], k.shape[-2], q.device
    mask = ~torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
    mask = repeat(mask, "n m -> b n m", b=b)
    return mask

class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features**-0.5
        self.num_heads = num_heads
        mid_features = head_features * num_heads
        out_features = default(out_features, features)

        self.to_out = nn.Linear(
            in_features=mid_features, out_features=out_features
        )

        self.use_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')

        if not self.use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # Use flash attention for A100 GPUs
            self.sdp_kernel_config = (True, False, False)
        else:
            # Don't use flash attention for other GPUs
            self.sdp_kernel_config = (False, True, True)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, is_causal: bool = False
    ) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        if not self.use_flash:
            if is_causal and not mask:
                # Mask out future tokens for causal attention
                mask = causal_mask(q, k)

            # Compute similarity matrix and add eventual mask
            sim = einsum("... n d, ... m d -> ... n m", q, k) * self.scale
            sim = add_mask(sim, mask) if exists(mask) else sim

            # Get attention matrix with softmax
            attn = sim.softmax(dim=-1, dtype=torch.float32)

            # Compute values
            out = einsum("... n m, ... m d -> ... n d", attn, v)
        else:
            with sdp_kernel(*self.sdp_kernel_config):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.context_features = context_features
        self.causal = causal
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            out_features=out_features,
        )

    def forward(
        self,
        x: Tensor, # [b, n, c]
        context: Optional[Tensor] = None, # [b, m, d]
        context_mask: Optional[Tensor] = None,  # [b, m], false is masked,
        causal: Optional[bool] = False,
    ) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)

        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        if exists(context_mask):
            # Mask out cross-attention for padding tokens
            mask = repeat(context_mask, "b m -> b m d", d=v.shape[-1])
            k, v = k * mask, v * mask

        # Compute and return attention
        return self.attention(q, k, v, is_causal=self.causal or causal)


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )

"""
Transformer Blocks
"""


class TransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = Attention(
            features=features,
            num_heads=num_heads,
            head_features=head_features
        )

        if self.use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None, context_mask: Optional[Tensor] = None, causal: Optional[bool] = False) -> Tensor:
        x = self.attention(x, causal=causal) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context, context_mask=context_mask) + x
        x = self.feed_forward(x) + x
        return x


"""
Transformers
"""


class Transformer1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_features: Optional[int] = None,
    ):
        super().__init__()

        self.to_in = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True),
            Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
            Rearrange("b c t -> b t c"),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    features=channels,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    context_features=context_features,
                )
                for i in range(num_layers)
            ]
        )

        self.to_out = nn.Sequential(
            Rearrange("b t c -> b c t"),
            Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None, context_mask: Optional[Tensor] = None, causal=False) -> Tensor:
        x = self.to_in(x)
        for block in self.blocks:
            x = block(x, context=context, context_mask=context_mask, causal=causal)
        x = self.to_out(x)
        return x


"""
Time Embeddings
"""


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device, half_dim = x.device, self.dim // 2
        emb = torch.tensor(log(10000) / (half_dim - 1), device=device)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


"""
Encoder/Decoder Components
"""


class DownsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_groups: int,
        num_layers: int,
        kernel_multiplier: int = 2,
        use_pre_downsample: bool = True,
        use_skip: bool = False,
        extract_channels: int = 0,
        context_channels: int = 0,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
    ):
        super().__init__()
        self.use_pre_downsample = use_pre_downsample
        self.use_skip = use_skip
        self.use_transformer = num_transformer_blocks > 0
        self.use_extract = extract_channels > 0
        self.use_context = context_channels > 0

        channels = out_channels if use_pre_downsample else in_channels

        self.downsample = Downsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            kernel_multiplier=kernel_multiplier,
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + context_channels if i == 0 else channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    context_mapping_features=context_mapping_features,
                )
                for i in range(num_layers)
            ]
        )

        if self.use_transformer:
            assert (
                (exists(attention_heads) or exists(attention_features))
                and exists(attention_multiplier)
            )

            if attention_features is None and attention_heads is not None:
                attention_features = channels // attention_heads

            if attention_heads is None and attention_features is not None:
                attention_heads = channels // attention_features

            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features
            )

        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups,
            )

    def forward(
        self,
        x: Tensor,
        *,
        mapping: Optional[Tensor] = None,
        channels: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False
    ) -> Union[Tuple[Tensor, List[Tensor]], Tensor]:

        if self.use_pre_downsample:
            x = self.downsample(x)

        if self.use_context and exists(channels):
            x = torch.cat([x, channels], dim=1)

        skips = []
        for block in self.blocks:
            x = block(x, mapping=mapping, causal=causal)
            skips += [x] if self.use_skip else []

        if self.use_transformer:
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)
            skips += [x] if self.use_skip else []

        if not self.use_pre_downsample:
            x = self.downsample(x)

        if self.use_extract:
            extracted = self.to_extracted(x)
            return x, extracted

        return (x, skips) if self.use_skip else x


class UpsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_layers: int,
        num_groups: int,
        use_nearest: bool = False,
        use_pre_upsample: bool = False,
        use_skip: bool = False,
        skip_channels: int = 0,
        use_skip_scale: bool = False,
        extract_channels: int = 0,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_extract = extract_channels > 0
        self.use_pre_upsample = use_pre_upsample
        self.use_transformer = num_transformer_blocks > 0
        self.use_skip = use_skip
        self.skip_scale = 2 ** -0.5 if use_skip_scale else 1.0

        channels = out_channels if use_pre_upsample else in_channels

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + skip_channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    context_mapping_features=context_mapping_features
                )
                for _ in range(num_layers)
            ]
        )

        if self.use_transformer:
            assert (
                (exists(attention_heads) or exists(attention_features))
                and exists(attention_multiplier)
            )

            if attention_features is None and attention_heads is not None:
                attention_features = channels // attention_heads

            if attention_heads is None and attention_features is not None:
                attention_heads = channels // attention_features

            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
            )

        self.upsample = Upsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            use_nearest=use_nearest,
        )

        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups
            )

    def add_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([x, skip * self.skip_scale], dim=1)

    def forward(
        self,
        x: Tensor,
        *,
        skips: Optional[List[Tensor]] = None,
        mapping: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:

        if self.use_pre_upsample:
            x = self.upsample(x)

        for block in self.blocks:
            x = self.add_skip(x, skip=skips.pop()) if exists(skips) else x
            x = block(x, mapping=mapping, causal=causal)

        if self.use_transformer:
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)

        if not self.use_pre_upsample:
            x = self.upsample(x)

        if self.use_extract:
            extracted = self.to_extracted(x)
            return x, extracted

        return x


class BottleneckBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_groups: int,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None
    ):
        super().__init__()
        self.use_transformer = num_transformer_blocks > 0

        self.pre_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            context_mapping_features=context_mapping_features
        )

        if self.use_transformer:
            assert (
                (exists(attention_heads) or exists(attention_features))
                and exists(attention_multiplier)
            )

            if attention_features is None and attention_heads is not None:
                attention_features = channels // attention_heads

            if attention_heads is None and attention_features is not None:
                attention_heads = channels // attention_features

            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
            )

        self.post_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            context_mapping_features=context_mapping_features
        )

    def forward(
        self,
        x: Tensor,
        *,
        mapping: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False
    ) -> Tensor:
        x = self.pre_block(x, mapping=mapping, causal=causal)
        if self.use_transformer:
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)
        x = self.post_block(x, mapping=mapping, causal=causal)
        return x


"""
UNet
"""


class UNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        attentions: Sequence[int],
        patch_size: int = 1,
        resnet_groups: int = 8,
        use_context_time: bool = True,
        kernel_multiplier_downsample: int = 2,
        use_nearest_upsample: bool = False,
        use_skip_scale: bool = True,
        use_stft: bool = False,
        use_stft_context: bool = False,
        out_channels: Optional[int] = None,
        context_features: Optional[int] = None,
        context_features_multiplier: int = 4,
        context_channels: Optional[Sequence[int]] = None,
        context_embedding_features: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        out_channels = default(out_channels, in_channels)
        context_channels = list(default(context_channels, []))
        num_layers = len(multipliers) - 1
        use_context_features = exists(context_features)
        use_context_channels = len(context_channels) > 0
        context_mapping_features = None

        attention_kwargs, kwargs = groupby("attention_", kwargs, keep_prefix=True)

        self.num_layers = num_layers
        self.use_context_time = use_context_time
        self.use_context_features = use_context_features
        self.use_context_channels = use_context_channels
        self.use_stft = use_stft
        self.use_stft_context = use_stft_context

        self.context_features = context_features
        context_channels_pad_length = num_layers + 1 - len(context_channels)
        context_channels = context_channels + [0] * context_channels_pad_length
        self.context_channels = context_channels
        self.context_embedding_features = context_embedding_features

        if use_context_channels:
            has_context = [c > 0 for c in context_channels]
            self.has_context = has_context
            self.channels_ids = [sum(has_context[:i]) for i in range(len(has_context))]

        assert (
            len(factors) == num_layers
            and len(attentions) >= num_layers
            and len(num_blocks) == num_layers
        )

        if use_context_time or use_context_features:
            context_mapping_features = channels * context_features_multiplier

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )

        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=channels, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(
                    in_features=context_features, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_stft:
            stft_kwargs, kwargs = groupby("stft_", kwargs)
            assert "num_fft" in stft_kwargs, "stft_num_fft required if use_stft=True"
            stft_channels = (stft_kwargs["num_fft"] // 2 + 1) * 2
            in_channels *= stft_channels
            out_channels *= stft_channels
            context_channels[0] *= stft_channels if use_stft_context else 1
            assert exists(in_channels) and exists(out_channels)
            self.stft = STFT(**stft_kwargs)

        assert not kwargs, f"Unknown arguments: {', '.join(list(kwargs.keys()))}"

        self.to_in = Patcher(
            in_channels=in_channels + context_channels[0],
            out_channels=channels * multipliers[0],
            patch_size=patch_size,
            context_mapping_features=context_mapping_features
        )

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    context_mapping_features=context_mapping_features,
                    context_channels=context_channels[i + 1],
                    context_embedding_features=context_embedding_features,
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    kernel_multiplier=kernel_multiplier_downsample,
                    num_groups=resnet_groups,
                    use_pre_downsample=True,
                    use_skip=True,
                    num_transformer_blocks=attentions[i],
                    **attention_kwargs,
                )
                for i in range(num_layers)
            ]
        )

        self.bottleneck = BottleneckBlock1d(
            channels=channels * multipliers[-1],
            context_mapping_features=context_mapping_features,
            context_embedding_features=context_embedding_features,
            num_groups=resnet_groups,
            num_transformer_blocks=attentions[-1],
            **attention_kwargs,
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i + 1],
                    out_channels=channels * multipliers[i],
                    context_mapping_features=context_mapping_features,
                    context_embedding_features=context_embedding_features,
                    num_layers=num_blocks[i] + (1 if attentions[i] else 0),
                    factor=factors[i],
                    use_nearest=use_nearest_upsample,
                    num_groups=resnet_groups,
                    use_skip_scale=use_skip_scale,
                    use_pre_upsample=False,
                    use_skip=True,
                    skip_channels=channels * multipliers[i + 1],
                    num_transformer_blocks=attentions[i],
                    **attention_kwargs,
                )
                for i in reversed(range(num_layers))
            ]
        )

        self.to_out = Unpatcher(
            in_channels=channels * multipliers[0],
            out_channels=out_channels,
            patch_size=patch_size,
            context_mapping_features=context_mapping_features
        )

    def get_channels(
        self, channels_list: Optional[Sequence[Tensor]] = None, layer: int = 0
    ) -> Optional[Tensor]:
        """Gets context channels at `layer` and checks that shape is correct"""
        use_context_channels = self.use_context_channels and self.has_context[layer]
        if not use_context_channels:
            return None
        assert exists(channels_list), "Missing context"
        # Get channels index (skipping zero channel contexts)
        channels_id = self.channels_ids[layer]
        # Get channels
        channels = channels_list[channels_id]
        message = f"Missing context for layer {layer} at index {channels_id}"
        assert exists(channels), message
        # Check channels
        num_channels = self.context_channels[layer]
        message = f"Expected context with {num_channels} channels at idx {channels_id}"
        assert channels.shape[1] == num_channels, message
        # STFT channels if requested
        channels = self.stft.encode1d(channels) if self.use_stft_context else channels  # type: ignore # noqa
        return channels

    def get_mapping(
        self, time: Optional[Tensor] = None, features: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]
        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = reduce(torch.stack(items), "n b m -> b m", "sum")
            mapping = self.to_mapping(mapping)
        return mapping

    def forward(
        self,
        x: Tensor,
        time: Optional[Tensor] = None,
        *,
        features: Optional[Tensor] = None,
        channels_list: Optional[Sequence[Tensor]] = None,
        embedding: Optional[Tensor] = None,
        embedding_mask: Optional[Tensor] = None,
        causal: Optional[bool] = False,
    ) -> Tensor:
        channels = self.get_channels(channels_list, layer=0)
        # Apply stft if required
        x = self.stft.encode1d(x) if self.use_stft else x  # type: ignore
        # Concat context channels at layer 0 if provided
        x = torch.cat([x, channels], dim=1) if exists(channels) else x
        # Compute mapping from time and features
        mapping = self.get_mapping(time, features)
        x = self.to_in(x, mapping, causal=causal)
        skips_list = [x]

        for i, downsample in enumerate(self.downsamples):
            channels = self.get_channels(channels_list, layer=i + 1)
            x, skips = downsample(
                x, mapping=mapping, channels=channels, embedding=embedding, embedding_mask=embedding_mask, causal=causal
            )
            skips_list += [skips]

        x = self.bottleneck(x, mapping=mapping, embedding=embedding, embedding_mask=embedding_mask, causal=causal)

        for i, upsample in enumerate(self.upsamples):
            skips = skips_list.pop()
            x = upsample(x, skips=skips, mapping=mapping, embedding=embedding, embedding_mask=embedding_mask, causal=causal)

        x += skips_list.pop()
        x = self.to_out(x, mapping, causal=causal)
        x = self.stft.decode1d(x) if self.use_stft else x

        return x


""" Conditioning Modules """


class FixedEmbedding(nn.Module):
    def __init__(self, max_length: int, features: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, length, device = *x.shape[0:2], x.device
        assert_message = "Input sequence length must be <= max_length"
        assert length <= self.max_length, assert_message
        position = torch.arange(length, device=device)
        fixed_embedding = self.embedding(position)
        fixed_embedding = repeat(fixed_embedding, "n d -> b n d", b=batch_size)
        return fixed_embedding


def rand_bool(shape: Any, proba: float, device: Any = None) -> Tensor:
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


class UNetCFG1d(UNet1d):

    """UNet1d with Classifier-Free Guidance"""

    def __init__(
        self,
        context_embedding_max_length: int,
        context_embedding_features: int,
        use_xattn_time: bool = False,
        **kwargs,
    ):
        super().__init__(
            context_embedding_features=context_embedding_features, **kwargs
        )

        self.use_xattn_time = use_xattn_time

        if use_xattn_time:
            assert exists(context_embedding_features)
            self.to_time_embedding = nn.Sequential(
                TimePositionalEmbedding(
                    dim=kwargs["channels"], out_features=context_embedding_features
                ),
                nn.GELU(),
            )

            context_embedding_max_length += 1   # Add one for time embedding

        self.fixed_embedding = FixedEmbedding(
            max_length=context_embedding_max_length, features=context_embedding_features
        )

    def forward(  # type: ignore
        self,
        x: Tensor,
        time: Tensor,
        *,
        embedding: Tensor,
        embedding_mask: Optional[Tensor] = None,
        embedding_scale: float = 1.0,
        embedding_mask_proba: float = 0.0,
        batch_cfg: bool = False,
        rescale_cfg: bool = False,
        scale_phi: float = 0.4,
        negative_embedding: Optional[Tensor] = None,
        negative_embedding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        b, device = embedding.shape[0], embedding.device

        if self.use_xattn_time:
            embedding = torch.cat([embedding, self.to_time_embedding(time).unsqueeze(1)], dim=1)

            if embedding_mask is not None:
                embedding_mask = torch.cat([embedding_mask, torch.ones((b, 1), device=device)], dim=1)

        fixed_embedding = self.fixed_embedding(embedding)

        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba, device=device
            )
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            if batch_cfg:
                batch_x = torch.cat([x, x], dim=0)
                batch_time = torch.cat([time, time], dim=0)

                if negative_embedding is not None:
                    if negative_embedding_mask is not None:
                        negative_embedding_mask = negative_embedding_mask.to(torch.bool).unsqueeze(2)

                        negative_embedding = torch.where(negative_embedding_mask, negative_embedding, fixed_embedding)
                    
                    batch_embed = torch.cat([embedding, negative_embedding], dim=0)

                else:
                    batch_embed = torch.cat([embedding, fixed_embedding], dim=0)

                batch_mask = None
                if embedding_mask is not None:
                    batch_mask = torch.cat([embedding_mask, embedding_mask], dim=0)

                batch_features = None
                features = kwargs.pop("features", None)
                if self.use_context_features:
                    batch_features = torch.cat([features, features], dim=0)

                batch_channels = None
                channels_list = kwargs.pop("channels_list", None)
                if self.use_context_channels:
                    batch_channels = []
                    for channels in channels_list:
                        batch_channels += [torch.cat([channels, channels], dim=0)]

                # Compute both normal and fixed embedding outputs
                batch_out = super().forward(batch_x, batch_time, embedding=batch_embed, embedding_mask=batch_mask, features=batch_features, channels_list=batch_channels, **kwargs)
                out, out_masked = batch_out.chunk(2, dim=0)
           
            else:
                # Compute both normal and fixed embedding outputs
                out = super().forward(x, time, embedding=embedding, embedding_mask=embedding_mask, **kwargs)
                out_masked = super().forward(x, time, embedding=fixed_embedding, embedding_mask=embedding_mask, **kwargs)

            out_cfg = out_masked + (out - out_masked) * embedding_scale

            if rescale_cfg:

                out_std = out.std(dim=1, keepdim=True)
                out_cfg_std = out_cfg.std(dim=1, keepdim=True)

                return scale_phi * (out_cfg * (out_std/out_cfg_std)) + (1-scale_phi) * out_cfg

            else:

                return out_cfg
                
        else:
            return super().forward(x, time, embedding=embedding, embedding_mask=embedding_mask, **kwargs)


class UNetNCCA1d(UNet1d):

    """UNet1d with Noise Channel Conditioning Augmentation"""

    def __init__(self, context_features: int, **kwargs):
        super().__init__(context_features=context_features, **kwargs)
        self.embedder = NumberEmbedder(features=context_features)

    def expand(self, x: Any, shape: Tuple[int, ...]) -> Tensor:
        x = x if torch.is_tensor(x) else torch.tensor(x)
        return x.expand(shape)

    def forward(  # type: ignore
        self,
        x: Tensor,
        time: Tensor,
        *,
        channels_list: Sequence[Tensor],
        channels_augmentation: Union[
            bool, Sequence[bool], Sequence[Sequence[bool]], Tensor
        ] = False,
        channels_scale: Union[
            float, Sequence[float], Sequence[Sequence[float]], Tensor
        ] = 0,
        **kwargs,
    ) -> Tensor:
        b, n = x.shape[0], len(channels_list)
        channels_augmentation = self.expand(channels_augmentation, shape=(b, n)).to(x)
        channels_scale = self.expand(channels_scale, shape=(b, n)).to(x)

        # Augmentation (for each channel list item)
        for i in range(n):
            scale = channels_scale[:, i] * channels_augmentation[:, i]
            scale = rearrange(scale, "b -> b 1 1")
            item = channels_list[i]
            channels_list[i] = torch.randn_like(item) * scale + item * (1 - scale)  # type: ignore # noqa

        # Scale embedding (sum reduction if more than one channel list item)
        channels_scale_emb = self.embedder(channels_scale)
        channels_scale_emb = reduce(channels_scale_emb, "b n d -> b d", "sum")

        return super().forward(
            x=x,
            time=time,
            channels_list=channels_list,
            features=channels_scale_emb,
            **kwargs,
        )


class UNetAll1d(UNetCFG1d, UNetNCCA1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):  # type: ignore
        return UNetCFG1d.forward(self, *args, **kwargs)


def XUNet1d(type: str = "base", **kwargs) -> UNet1d:
    if type == "base":
        return UNet1d(**kwargs)
    elif type == "all":
        return UNetAll1d(**kwargs)
    elif type == "cfg":
        return UNetCFG1d(**kwargs)
    elif type == "ncca":
        return UNetNCCA1d(**kwargs)
    else:
        raise ValueError(f"Unknown XUNet1d type: {type}")

class NumberEmbedder(nn.Module):
    def __init__(
        self,
        features: int,
        dim: int = 256,
    ):
        super().__init__()
        self.features = features
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: Union[List[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        x = embedding.view(*shape, self.features)
        return x  # type: ignore


"""
Audio Transforms
"""


class STFT(nn.Module):
    """Helper for torch stft and istft"""

    def __init__(
        self,
        num_fft: int = 1023,
        hop_length: int = 256,
        window_length: Optional[int] = None,
        length: Optional[int] = None,
        use_complex: bool = False,
    ):
        super().__init__()
        self.num_fft = num_fft
        self.hop_length = default(hop_length, floor(num_fft // 4))
        self.window_length = default(window_length, num_fft)
        self.length = length
        self.register_buffer("window", torch.hann_window(self.window_length))
        self.use_complex = use_complex

    def encode(self, wave: Tensor) -> Tuple[Tensor, Tensor]:
        b = wave.shape[0]
        wave = rearrange(wave, "b c t -> (b c) t")

        stft = torch.stft(
            wave,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            return_complex=True,
            normalized=True,
        )

        if self.use_complex:
            # Returns real and imaginary
            stft_a, stft_b = stft.real, stft.imag
        else:
            # Returns magnitude and phase matrices
            magnitude, phase = torch.abs(stft), torch.angle(stft)
            stft_a, stft_b = magnitude, phase

        return rearrange_many((stft_a, stft_b), "(b c) f l -> b c f l", b=b)

    def decode(self, stft_a: Tensor, stft_b: Tensor) -> Tensor:
        b, l = stft_a.shape[0], stft_a.shape[-1]  # noqa
        length = closest_power_2(l * self.hop_length)

        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b c f l -> (b c) f l")

        if self.use_complex:
            real, imag = stft_a, stft_b
        else:
            magnitude, phase = stft_a, stft_b
            real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)

        stft = torch.stack([real, imag], dim=-1)

        wave = torch.istft(
            stft,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            length=default(self.length, length),
            normalized=True,
        )

        return rearrange(wave, "(b c) t -> b c t", b=b)

    def encode1d(
        self, wave: Tensor, stacked: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        stft_a, stft_b = self.encode(wave)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b c f l -> b (c f) l")
        return torch.cat((stft_a, stft_b), dim=1) if stacked else (stft_a, stft_b)

    def decode1d(self, stft_pair: Tensor) -> Tensor:
        f = self.num_fft // 2 + 1
        stft_a, stft_b = stft_pair.chunk(chunks=2, dim=1)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b (c f) l -> b c f l", f=f)
        return self.decode(stft_a, stft_b)

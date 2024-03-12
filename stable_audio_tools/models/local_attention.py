import torch

from einops import rearrange
from torch import nn

from .blocks import AdaRMSNorm
from .transformer import Attention, FeedForward, RotaryEmbedding, LayerNorm

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

# Adapted from https://github.com/lucidrains/local-attention/blob/master/local_attention/transformer.py
class ContinuousLocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_in = None,
        dim_out = None,
        causal = False,
        local_attn_window_size = 64,
        heads = 8,
        ff_mult = 2,
        cond_dim = 0,
        cross_attn_cond_dim = 0,
        **kwargs
    ):
        super().__init__()
        
        dim_head = dim//heads

        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim) if dim_in is not None else nn.Identity()

        self.project_out = nn.Linear(dim, dim_out) if dim_out is not None else nn.Identity()

        self.local_attn_window_size = local_attn_window_size

        self.cond_dim = cond_dim

        self.cross_attn_cond_dim = cross_attn_cond_dim

        self.rotary_pos_emb = RotaryEmbedding(max(dim_head // 2, 32))
       
        for _ in range(depth):

            self.layers.append(nn.ModuleList([
                AdaRMSNorm(dim, cond_dim, eps=1e-8) if cond_dim > 0 else LayerNorm(dim),
                Attention(
                    dim=dim,
                    dim_heads=dim_head,
                    causal=causal,
                    zero_init_output=True,
                    natten_kernel_size=local_attn_window_size,
                ),
                Attention(
                    dim=dim,
                    dim_heads=dim_head,
                    dim_context = cross_attn_cond_dim,
                    zero_init_output=True
                ) if self.cross_attn_cond_dim > 0 else nn.Identity(),
                AdaRMSNorm(dim, cond_dim, eps=1e-8) if cond_dim > 0 else LayerNorm(dim),
                FeedForward(dim = dim, mult = ff_mult, no_bias=True)
            ]))

    def forward(self, x, mask = None, cond = None, cross_attn_cond = None, cross_attn_cond_mask = None, prepend_cond = None):
 
        x = checkpoint(self.project_in, x)

        if prepend_cond is not None:
            x = torch.cat([prepend_cond, x], dim=1)

        pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])

        for attn_norm, attn, xattn, ff_norm, ff in self.layers:

            residual = x
            if cond is not None:
                x = checkpoint(attn_norm, x, cond)
            else:
                x = checkpoint(attn_norm, x)

            x = checkpoint(attn, x, mask = mask, rotary_pos_emb=pos_emb) + residual

            if cross_attn_cond is not None:
                x = checkpoint(xattn, x, context=cross_attn_cond, context_mask=cross_attn_cond_mask) + x

            residual = x

            if cond is not None:
                x = checkpoint(ff_norm, x, cond)
            else:
                x = checkpoint(ff_norm, x)

            x = checkpoint(ff, x) + residual

        return checkpoint(self.project_out, x)

class TransformerDownsampleBlock1D(nn.Module):
    def __init__(
        self, 
        in_channels,
        embed_dim = 768,
        depth = 3,
        heads = 12,
        downsample_ratio = 2,
        local_attn_window_size = 64,
        **kwargs
    ):
        super().__init__()

        self.downsample_ratio = downsample_ratio

        self.transformer = ContinuousLocalTransformer(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            local_attn_window_size=local_attn_window_size,
            **kwargs
        )

        self.project_in = nn.Linear(in_channels, embed_dim, bias=False) if in_channels != embed_dim else nn.Identity()

        self.project_down = nn.Linear(embed_dim * self.downsample_ratio, embed_dim, bias=False)
        
    
    def forward(self, x):

        x = checkpoint(self.project_in, x)

        # Compute
        x = self.transformer(x)

        # Trade sequence length for channels
        x = rearrange(x, "b (n r) c -> b n (c r)", r=self.downsample_ratio)

        # Project back to embed dim
        x = checkpoint(self.project_down, x)

        return x

class TransformerUpsampleBlock1D(nn.Module):
    def __init__(
        self, 
        in_channels,
        embed_dim,
        depth = 3,
        heads = 12,
        upsample_ratio = 2,
        local_attn_window_size = 64,
        **kwargs
    ):
        super().__init__()

        self.upsample_ratio = upsample_ratio

        self.transformer = ContinuousLocalTransformer(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            local_attn_window_size = local_attn_window_size,
            **kwargs
        )

        self.project_in = nn.Linear(in_channels, embed_dim, bias=False) if in_channels != embed_dim else nn.Identity()

        self.project_up = nn.Linear(embed_dim, embed_dim * self.upsample_ratio, bias=False)
        
    def forward(self, x):

        # Project to embed dim
        x = checkpoint(self.project_in, x)

        # Project to increase channel dim
        x = checkpoint(self.project_up, x)

        # Trade channels for sequence length
        x = rearrange(x, "b n (c r) -> b (n r) c", r=self.upsample_ratio)

        # Compute
        x = self.transformer(x)        

        return x
   

class TransformerEncoder1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dims = [96, 192, 384, 768],
        heads = [12, 12, 12, 12],
        depths = [3, 3, 3, 3],
        ratios = [2, 2, 2, 2],
        local_attn_window_size = 64,
        **kwargs
    ):
        super().__init__()
        
        layers = []
       
        for layer in range(len(depths)):
            prev_dim = embed_dims[layer - 1] if layer > 0 else embed_dims[0]

            layers.append(
                TransformerDownsampleBlock1D(
                    in_channels = prev_dim,
                    embed_dim = embed_dims[layer],
                    heads = heads[layer],
                    depth = depths[layer],
                    downsample_ratio = ratios[layer],
                    local_attn_window_size = local_attn_window_size,
                    **kwargs
                )
            )
        
        self.layers = nn.Sequential(*layers)

        self.project_in = nn.Linear(in_channels, embed_dims[0], bias=False)
        self.project_out = nn.Linear(embed_dims[-1], out_channels, bias=False)

    def forward(self, x):
        x = rearrange(x, "b c n -> b n c")
        x = checkpoint(self.project_in, x)
        x = self.layers(x)
        x = checkpoint(self.project_out, x)
        x = rearrange(x, "b n c -> b c n")

        return x


class TransformerDecoder1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dims = [768, 384, 192, 96],
        heads = [12, 12, 12, 12],
        depths = [3, 3, 3, 3],
        ratios = [2, 2, 2, 2],
        local_attn_window_size = 64,
        **kwargs
    ):

        super().__init__()

        layers = []
       
        for layer in range(len(depths)):
            prev_dim = embed_dims[layer - 1] if layer > 0 else embed_dims[0]

            layers.append(
                TransformerUpsampleBlock1D(
                    in_channels = prev_dim,
                    embed_dim = embed_dims[layer],
                    heads = heads[layer],
                    depth = depths[layer],
                    upsample_ratio = ratios[layer],
                    local_attn_window_size = local_attn_window_size,
                    **kwargs
                )
            )
        
        self.layers = nn.Sequential(*layers)

        self.project_in = nn.Linear(in_channels, embed_dims[0], bias=False)
        self.project_out = nn.Linear(embed_dims[-1], out_channels, bias=False)

    def forward(self, x):
        x = rearrange(x, "b c n -> b n c")
        x = checkpoint(self.project_in, x)
        x = self.layers(x)
        x = checkpoint(self.project_out, x)
        x = rearrange(x, "b n c -> b c n")
        return x
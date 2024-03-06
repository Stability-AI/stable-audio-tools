import torch
from torch import nn
from x_transformers import ContinuousTransformerWrapper, Decoder

from .transformer import ContinuousTransformer

# Interface for backbone of a language model
# Handles conditioning and cross-attention
# Does not have to deal with patterns or quantizer heads
class AudioLMBackbone(nn.Module):
    def __init__(self, embed_dim: int, use_generation_cache=False, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_generation_cache = use_generation_cache

    def forward(
        self, 
        x, 
        cross_attn_cond=None, 
        prepend_cond=None, 
        prepend_cond_mask=None,
        global_cond=None,
        use_cache=False,
        **kwargs
        ):
        raise NotImplementedError
    
    def reset_generation_cache(
        self,
        max_seq_len, 
        batch_size,
        dtype=None
    ):
        pass

    def update_generation_cache(
        self,
        seqlen_offset
    ):
        pass

class XTransformersAudioLMBackbone(AudioLMBackbone):
    def __init__(self,
                 embed_dim: int,
                 cross_attn_cond_dim: int = 0,
                 prepend_cond_dim: int = 0,
                 **kwargs):
        super().__init__(embed_dim=embed_dim)

        # Embeddings are done in the AudioLanguageModel, so we use the continuous-input transformer
        self.model = ContinuousTransformerWrapper(
            dim_in=embed_dim,
            dim_out=embed_dim,
            max_seq_len=0, #Not relevant without absolute positional embeds,
            attn_layers=Decoder(
                dim=embed_dim,
                attn_flash = True,
                cross_attend = cross_attn_cond_dim > 0,
                zero_init_branch_output=True,
                use_abs_pos_emb = False,
                rotary_pos_emb=True,
                ff_swish = True,
                ff_glu = True,
                **kwargs
            )
        )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if cross_attn_cond_dim > 0:
            # Cross-attention conditioning
            self.to_cross_attn_embed = nn.Sequential(
                nn.Linear(cross_attn_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

    def forward(self, x, mask=None, prepend_cond=None, prepend_cond_mask=None, cross_attn_cond=None, global_cond=None, use_cache=False):

        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)
            prepend_length = prepend_cond.shape[1]

            if prepend_cond_mask is not None:
                # Cast mask to bool
                prepend_cond_mask = prepend_cond_mask.bool()

        if cross_attn_cond is not None:
            # Project the cross-attention conditioning to the embedding dimension
            cross_attn_cond = self.to_cross_attn_embed(cross_attn_cond)

        return self.model(x, mask=mask, context=cross_attn_cond, prepend_embeds=prepend_cond, prepend_mask=prepend_cond_mask)[:, prepend_length:, :]
    
class ContinuousTransformerAudioLMBackbone(AudioLMBackbone):
    def __init__(self,
                 embed_dim: int,
                 cross_attn_cond_dim: int = 0,
                 prepend_cond_dim: int = 0,
                 project_cross_attn_cond: bool = False,
                 **kwargs):
        super().__init__(embed_dim=embed_dim)

        # Embeddings are done in the AudioLanguageModel, so we use the continuous-input transformer
        self.model = ContinuousTransformer(
            dim=embed_dim,
            dim_in=embed_dim,
            dim_out=embed_dim,
            cross_attend = cross_attn_cond_dim > 0,
            cond_token_dim = embed_dim if project_cross_attn_cond else cross_attn_cond_dim,
            causal=True,
            **kwargs
        )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if cross_attn_cond_dim > 0 and project_cross_attn_cond:
            # Cross-attention conditioning
            self.to_cross_attn_embed = nn.Sequential(
                nn.Linear(cross_attn_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )
        else:
            self.to_cross_attn_embed = nn.Identity()

    def forward(self, x, mask=None, prepend_cond=None, prepend_cond_mask=None, cross_attn_cond=None, global_cond=None, use_cache=False):

        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)
            prepend_length = prepend_cond.shape[1]

            if prepend_cond_mask is not None:
                # Cast mask to bool
                prepend_cond_mask = prepend_cond_mask.bool()

        if cross_attn_cond is not None:
            # Project the cross-attention conditioning to the embedding dimension
            cross_attn_cond = self.to_cross_attn_embed(cross_attn_cond)

        return self.model(x, mask=mask, context=cross_attn_cond, prepend_embeds=prepend_cond, prepend_mask=prepend_cond_mask)[:, prepend_length:, :]

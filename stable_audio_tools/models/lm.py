from dataclasses import dataclass
import torch
from tqdm.auto import trange
import typing as tp
from einops import rearrange
from torch import nn

from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from .factory import create_pretransform_from_config
from .lm_backbone import AudioLMBackbone, XTransformersAudioLMBackbone, ContinuousTransformerAudioLMBackbone
from .pretransforms import Pretransform, AutoencoderPretransform, PretrainedDACPretransform, AudiocraftCompressionPretransform
from .utils import multinomial, sample_top_k, sample_top_p

from .codebook_patterns import (
    CodebooksPatternProvider,
    DelayedPatternProvider,
    MusicLMPattern,
    ParallelPatternProvider,
    UnrolledPatternProvider
)

# Copied and modified from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/lm.py under MIT license
# License can be found in LICENSES/LICENSE_META.txt

@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]

# Wrapper for a multi-codebook language model
# Handles patterns and quantizer heads
class AudioLanguageModel(nn.Module):
    def __init__(
            self, 
            pattern_provider: CodebooksPatternProvider, 
            backbone: AudioLMBackbone,
            num_quantizers: int,
            codebook_size: int
        ):
        super().__init__()

        self.pattern_provider = pattern_provider
        self.backbone = backbone
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        self.masked_token_id = codebook_size

        # Per-quantizer embedders
        # Add one for the mask embed
        self.embeds = nn.ModuleList([nn.Embedding(codebook_size + 1, backbone.embed_dim) for _ in range(num_quantizers)])

        # Per-quantizer output heads
        self.quantizer_heads = nn.ModuleList([
            nn.Linear(backbone.embed_dim, codebook_size) for _ in range(num_quantizers)
        ])

    def forward(self,
            sequence: torch.Tensor, #[batch, seq_len, 
            prepend_cond=None, #[batch, seq, channels]
            prepend_cond_mask=None,
            cross_attn_cond=None, #[batch, seq, channels],
            **kwargs
        ):

        batch, num_quantizers, seq_len = sequence.shape

        assert num_quantizers == self.num_quantizers, "Number of quantizers in sequence must match number of quantizers in model"

        backbone_input = sum([self.embeds[i](sequence[:, i]) for i in range(num_quantizers)]) # [batch, seq_len, embed_dim]

        dtype = next(self.parameters()).dtype

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(dtype)

            if prepend_cond_mask is not None:
                prepend_cond_mask = prepend_cond_mask.to(dtype)
            
        backbone_input = backbone_input.to(dtype)

        output = self.backbone(
            backbone_input,
            cross_attn_cond=cross_attn_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            **kwargs
        ) # [batch, seq_len, embed_dim]

        # Run output through quantizer heads
        logits = torch.stack([self.quantizer_heads[i](output) for i in range(num_quantizers)], dim=1) # [batch, num_quantizers, seq_len, codebook_size]

        return logits
    
    def compute_logits(
            self, 
            codes, #[batch, num_quantizers, seq_len]
            **kwargs):
        """
        Compute logits for a batch of codes, optionally conditioning on cross-attention and prepend conditioning
        Handles translation between input sequence and pattern-shifted sequence
        Only used during training
        """
        
        batch, _, seq_len = codes.shape

        pattern = self.pattern_provider.get_pattern(seq_len)

        # Apply the token pattern to the codes, shifting the codes as needed and masking out invalid steps
        shifted_codes, _, _ = pattern.build_pattern_sequence(
            codes,
            self.masked_token_id,
            keep_only_valid_steps=True
        )

        # Run the model to get logits for each quantizer [batch, num_quantizers, seq_len, codebook_size]
        logits = self(shifted_codes, **kwargs)

        # Rearrange logits to prepare to revert pattern
        logits = rearrange(logits, "b n s c -> b c n s")

        # Revert sequence logits back to original sequence length, removing masked steps
        logits, _, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=True
        )

        logits = rearrange(logits, "b c n t -> b n t c")

        logits_mask = logits_mask[None, :, :].expand(batch, -1, -1) # [batch, num_quantizers, seq_len]

        return LMOutput(logits=logits, mask=logits_mask)

# Conditioning and generation wrapper for a multi-codebook language model
# Handles conditioning, CFG, generation, and encoding/decoding
class AudioLanguageModelWrapper(nn.Module):
    def __init__(
            self, 
            pretransform: Pretransform,
            lm: AudioLanguageModel,
            sample_rate: int,
            min_input_length: int,
            conditioner: MultiConditioner = None,
            cross_attn_cond_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = []
        ):
        super().__init__()
        
        assert pretransform.is_discrete, "Pretransform must be discrete"
        self.pretransform = pretransform

        self.pretransform.requires_grad_(False)
        self.pretransform.eval()

        if isinstance(self.pretransform, AutoencoderPretransform):
            self.num_quantizers = self.pretransform.model.bottleneck.num_quantizers
            self.codebook_size = self.pretransform.model.bottleneck.codebook_size
        elif isinstance(self.pretransform, PretrainedDACPretransform):
            self.num_quantizers = self.pretransform.model.num_quantizers
            self.codebook_size = self.pretransform.model.codebook_size
        elif isinstance(self.pretransform, AudiocraftCompressionPretransform):
            self.num_quantizers = self.pretransform.num_quantizers
            self.codebook_size = self.pretransform.codebook_size
        else:
            raise NotImplementedError(f"Unrecognized pretransform type {type(self.pretransform)}")

        self.conditioner = conditioner

        self.lm = lm

        self.sample_rate = sample_rate
        self.min_input_length = min_input_length

        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.global_cond_ids = global_cond_ids
    
    def get_conditioning_inputs(self, cond: tp.Dict[str, tp.Any], negative=False):
        cross_attention_input = None
        prepend_cond = None
        prepend_cond_mask = None
        global_cond = None

        if len(self.cross_attn_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_input = torch.cat([cond[key][0] for key in self.cross_attn_cond_ids], dim=1)

        if len(self.prepend_cond_ids) > 0:
            # Concatenate all prepend conditioning inputs over the sequence dimension
            # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
            prepend_cond = torch.cat([cond[key][0] for key in self.prepend_cond_ids], dim=1)
            prepend_cond_mask = torch.cat([cond[key][1] for key in self.prepend_cond_ids], dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_cond = torch.cat([cond[key][0] for key in self.global_cond_ids], dim=-1)
            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_prepend_cond": prepend_cond,
                "negative_prepend_cond_mask": prepend_cond_mask,
                "negative_global_cond": global_cond
            }
        else:
            return {
                "cross_attn_cond": cross_attention_input,
                "prepend_cond": prepend_cond,
                "prepend_cond_mask": prepend_cond_mask,
                "global_cond": global_cond
            }
        
    def compute_logits(
            self, 
            codes, 
            condition_tensors=None, 
            cfg_dropout_prob=0.0,
            **kwargs
        ):
        """
        Compute logits for a batch of codes, and translates from conditioning inputs to model inputs
        Handles CFG dropout
        """

        if condition_tensors is None:
            condition_tensors = {}

        conditioning_inputs = self.get_conditioning_inputs(condition_tensors)

        cross_attn_cond = conditioning_inputs["cross_attn_cond"]
        prepend_cond = conditioning_inputs["prepend_cond"]
        prepend_cond_mask = conditioning_inputs["prepend_cond_mask"]
        global_cond = conditioning_inputs["global_cond"]

        if cfg_dropout_prob > 0.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)
        
            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

            if global_cond is not None:
                null_embed = torch.zeros_like(global_cond, device=global_cond.device)
                dropout_mask = torch.bernoulli(torch.full((global_cond.shape[0], 1), cfg_dropout_prob, device=global_cond.device)).to(torch.bool)
                global_cond = torch.where(dropout_mask, null_embed, global_cond)

        return self.lm.compute_logits(codes, cross_attn_cond=cross_attn_cond, prepend_cond=prepend_cond, prepend_cond_mask=prepend_cond_mask, global_cond=global_cond, **kwargs)
    
    def _sample_next_token(
            self, 
            sequence, #[batch, num_quantizers, seq_len]
            conditioning_tensors=None, 
            cross_attn_use_cfg=True,
            prepend_use_cfg=True,
            global_use_cfg=True,
            cfg_scale=1.0,
            top_k=250,
            top_p=0.0,
            temp=1.0,
            **kwargs
        ):
        """
        Sample the next token for a batch of codes, and translates from conditioning inputs to model inputs
        Handles CFG inference
        """

        if conditioning_tensors is None:
            conditioning_tensors = {}

        conditioning_inputs = self.get_conditioning_inputs(conditioning_tensors)

        cross_attn_cond = conditioning_inputs["cross_attn_cond"]
        prepend_cond = conditioning_inputs["prepend_cond"]
        prepend_cond_mask = conditioning_inputs["prepend_cond_mask"]
        global_cond = conditioning_inputs["global_cond"]

        if cfg_scale != 1.0:
            
            # Batch size is doubled to account for negative samples
            sequence = torch.cat([sequence, sequence], dim=0)

            if cross_attn_cond is not None and cross_attn_use_cfg:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                cross_attn_cond = torch.cat([cross_attn_cond, null_embed], dim=0)
            
            if prepend_cond is not None and prepend_use_cfg:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                prepend_cond = torch.cat([prepend_cond, null_embed], dim=0) 

                if prepend_cond_mask is not None:
                    prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)

            if global_cond is not None and global_use_cfg:
                null_embed = torch.zeros_like(global_cond, device=global_cond.device)

                global_cond = torch.cat([global_cond, null_embed], dim=0)

        logits = self.lm(sequence, cross_attn_cond=cross_attn_cond, prepend_cond=prepend_cond, prepend_cond_mask=prepend_cond_mask, global_cond=global_cond, **kwargs)

        if cfg_scale != 1.0:
            cond_logits, uncond_logits = logits.chunk(2, dim=0)

            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

        logits = rearrange(logits, "b n s c -> b n c s") # [batch, num_quantizers, codebook_size, seq_len]
        
        # Grab the logits for the last step
        logits = logits[:, :, :, -1] # [batch, num_quantizers, codebook_size]

        # Apply top-k or top-p sampling

        if temp > 0:
            probs = torch.softmax(logits / temp, dim=-1)

            if top_p > 0.0:
                next_token = sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = sample_top_k(probs, k=top_k)
            else:
                next_token = multinomial(probs, num_samples=1)

        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True) # [batch, num_quantizers, 1]

        return next_token

    @torch.no_grad()
    def generate(
        self,
        max_gen_len: int = 256,
        batch_size: tp.Optional[int] = None,
        init_data: tp.Optional[torch.Tensor] = None,
        conditioning: tp.Optional[tp.Dict[str, tp.Any]] = None,
        conditioning_tensors: tp.Optional[tp.Dict[str, tp.Any]] = None,
        callback: tp.Optional[tp.Callable[[int, int], None]] = None,
        use_cache: bool = True,
        cfg_scale: float = 1.0,
        **kwargs
    ):
        device = next(self.parameters()).device

        if conditioning_tensors is None and conditioning is not None:
            # Convert conditioning inputs to conditioning tensors
            conditioning_tensors = self.conditioner(conditioning, device)

        # Check that batch size is consistent across inputs
        possible_batch_sizes = []

        if batch_size is not None:
            possible_batch_sizes.append(batch_size)
        elif init_data is not None:
            possible_batch_sizes.append(init_data.shape[0])
        elif conditioning_tensors is not None:
            # Assume that the first conditioning tensor has the batch dimension
            possible_batch_sizes.append(conditioning_tensors[list(conditioning_tensors.keys())[0]][0].shape[0])
        else:
            possible_batch_sizes.append(1)

        assert [x == possible_batch_sizes[0] for x in possible_batch_sizes], "Batch size must be consistent across inputs"

        batch_size = possible_batch_sizes[0]
        
        if init_data is None:
            # Initialize with zeros
            assert batch_size > 0
            init_data = torch.zeros((batch_size, self.num_quantizers, 0), device=device, dtype=torch.long)

        batch_size, num_quantizers, seq_len = init_data.shape

        start_offset = seq_len
        assert start_offset < max_gen_len, "init data longer than max gen length"

        pattern = self.lm.pattern_provider.get_pattern(max_gen_len)

        unknown_token = -1

        # Initialize the generated codes with the init data, padded with unknown tokens
        gen_codes = torch.full((batch_size, num_quantizers, max_gen_len), unknown_token, device=device, dtype=torch.long)
        gen_codes[:, :, :start_offset] = init_data # [batch, num_quantizers, max_gen_len]

        gen_sequence, _, mask = pattern.build_pattern_sequence(gen_codes, self.lm.masked_token_id) # [batch, num_quantizers, gen_sequence_len]

        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        # Generation
        prev_offset = 0
        gen_sequence_len = gen_sequence.shape[-1]

        # Reset generation cache
        if use_cache and self.lm.backbone.use_generation_cache:
            self.lm.backbone.reset_generation_cache(max_gen_len, batch_size if cfg_scale == 1.0 else batch_size * 2)

        for offset in trange(start_offset_sequence, gen_sequence_len):

            # Get the full sequence up to the current offset
            curr_sequence = gen_sequence[..., prev_offset:offset]

            next_token = self._sample_next_token(
                curr_sequence,
                conditioning_tensors=conditioning_tensors,
                use_cache=use_cache,
                cfg_scale=cfg_scale,
                **kwargs
            )

            valid_mask = mask[..., offset:offset+1].expand(batch_size, -1, -1)
            next_token[~valid_mask] = self.lm.masked_token_id

            # Update the generated sequence with the next token
            gen_sequence[..., offset:offset+1] = torch.where(
                gen_sequence[..., offset:offset+1] == unknown_token,
                next_token, 
                gen_sequence[..., offset:offset+1]
            )

            if use_cache and self.lm.backbone.use_generation_cache:
                # Only update the offset if caching is being used
                prev_offset = offset

                self.lm.backbone.update_generation_cache(offset)

            if callback is not None:
                # Callback to report progress
                # Pass in the offset relative to the start of the sequence, and the length of the current sequence
                callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)

        assert not (gen_sequence == unknown_token).any(), "Unknown tokens in generated sequence"

        out_codes, _, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)

        # sanity checks over the returned codes and corresponding masks
        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()

        #out_codes = out_codes[..., 0:max_gen_len]

        return out_codes
    

    def generate_audio(
        self,
        **kwargs
    ):
        """
        Generate audio from a batch of codes
        """

        codes = self.generate(**kwargs)

        audio = self.pretransform.decode_tokens(codes)

        return audio


def create_audio_lm_from_config(config):
    model_config = config.get('model', None)
    assert model_config is not None, 'model config must be specified in config'

    sample_rate = config.get('sample_rate', None)
    assert sample_rate is not None, "Must specify sample_rate in config"
    
    lm_config = model_config.get('lm', None)
    assert lm_config is not None, 'lm config must be specified in model config'

    codebook_pattern = lm_config.get("codebook_pattern", "delay")

    pattern_providers = {
        'parallel': ParallelPatternProvider,
        'delay': DelayedPatternProvider,
        'unroll': UnrolledPatternProvider,
        'musiclm': MusicLMPattern,
    }

    pretransform_config = model_config.get("pretransform", None)
    
    pretransform = create_pretransform_from_config(pretransform_config, sample_rate)

    assert pretransform.is_discrete, "Pretransform must be discrete"

    min_input_length = pretransform.downsampling_ratio

    pattern_provider = pattern_providers[codebook_pattern](n_q=pretransform.num_quantizers)

    conditioning_config = model_config.get('conditioning', None)

    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)

    cross_attn_cond_ids = lm_config.get('cross_attention_cond_ids', [])
    prepend_cond_ids = lm_config.get('prepend_cond_ids', [])
    global_cond_ids = lm_config.get('global_cond_ids', [])

    lm_type = lm_config.get("type", None)
    lm_model_config = lm_config.get("config", None)

    assert lm_type is not None, "Must specify lm type in lm config"
    assert lm_model_config is not None, "Must specify lm model config in lm config"

    if lm_type == "x-transformers":
        backbone = XTransformersAudioLMBackbone(**lm_model_config)
    elif lm_type == "continuous_transformer":
        backbone = ContinuousTransformerAudioLMBackbone(**lm_model_config)
    else:
        raise NotImplementedError(f"Unrecognized lm type {lm_type}")

    lm = AudioLanguageModel(
        pattern_provider=pattern_provider,
        backbone=backbone,
        num_quantizers=pretransform.num_quantizers,
        codebook_size=pretransform.codebook_size
    )

    model = AudioLanguageModelWrapper(
        pretransform=pretransform,
        lm=lm,
        conditioner=conditioner,
        sample_rate=sample_rate,
        min_input_length=min_input_length,
        cross_attn_cond_ids=cross_attn_cond_ids,
        prepend_cond_ids=prepend_cond_ids,
        global_cond_ids=global_cond_ids
    )

    return model
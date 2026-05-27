import numpy as np
import torch
import typing as tp
import math
import copy
from torch.nn.functional import interpolate

from .utils import prepare_audio
from .sampling import sample_diffusion
from .inversion import invert_audio
from ..verbose import vprint


def generate_diffusion_uncond(
        model,
        steps: int = 250,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor:
    
    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    vprint(f"seed: {seed}")
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)
    # Extract sampler_type from sampler_kwargs if present
    sampler_type = sampler_kwargs.pop("sampler_type", "dpmpp-2m-sde")

    sampled = sample_diffusion(
        model=model.model,
        noise=noise,
        cond_inputs={},
        diffusion_objective=model.diffusion_objective,
        steps=steps,
        cfg_scale=1.0,
        pretransform=model.pretransform,
        # Sampler options
        sampler_type=sampler_type,
        batch_cfg=False,
        # Init data
        init_data=init_audio,
        init_noise_level=init_noise_level,
        # Other
        decode=not return_latents,
        **sampler_kwargs
    )

    return sampled


def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        inversion_params: dict = None,
        adapt_duration_to_conditioning: tp.Optional[bool] = None,
        duration_padding_sec: float = 6.0,
        use_effective_length_for_schedule: tp.Optional[bool] = None,
        mask_padding_attention: tp.Optional[bool] = None,
        apg_scale: float = 1.0,
        dist_shift = None,
        **sampler_kwargs
        ) -> torch.Tensor:
    """
    Generate audio from a prompt using a diffusion model.

    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        adapt_duration_to_conditioning: Adapt sample size based on seconds_total + duration_padding_sec.
            None (default) = auto-detect from model; True/False = explicit override.
        duration_padding_sec: Extra seconds to add when adapting duration (default 6.0).
        use_effective_length_for_schedule: Use effective_seq_len for distribution shift.
            None (default) = auto-detect from model; True/False = explicit override.
        mask_padding_attention: Create padding_mask for attention based on effective_seq_len.
            None (default) = auto-detect from model; True/False = explicit override.
        apg_scale: APG (Adaptive Projected Guidance) scale. 1.0 = full APG, 0.0 = vanilla CFG.
        dist_shift: Optional distribution shift override for sampling. If None, uses model.sampling_dist_shift.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.
    """

    if mask_padding_attention is None:
        mask_padding_attention = getattr(model, 'mask_padding_attention', False)
    if use_effective_length_for_schedule is None:
        use_effective_length_for_schedule = getattr(model, 'use_effective_length_for_schedule', False)
    if adapt_duration_to_conditioning is None:
        adapt_duration_to_conditioning = mask_padding_attention or use_effective_length_for_schedule

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # Optionally adapt sample size based on seconds_total conditioning
    if adapt_duration_to_conditioning and conditioning is not None:
        # Find the maximum seconds_total across the batch
        max_seconds = 0.0
        for cond_dict in conditioning:
            if "seconds_total" in cond_dict:
                max_seconds = max(max_seconds, cond_dict["seconds_total"])
        
        if max_seconds > 0:
            # Calculate target audio samples with padding, capped at original sample_size
            target_audio_samples = int((max_seconds + duration_padding_sec) * model.sample_rate)
            
            # Ensure we align to pretransform downsampling ratio if applicable
            if model.pretransform is not None:
                # Round up to nearest multiple of downsampling ratio, and also align to chunk size if using chunked attention
                ds_ratio = model.pretransform.downsampling_ratio
                latent_align = 1
                # Only perform if the encoder has chunked attention layers and not sliding window 
                if (hasattr(model.pretransform, 'model') and
                        hasattr(model.pretransform.model, 'encoder')):
                    encoder = model.pretransform.model.encoder
                    if hasattr(encoder, 'layers'):
                        first_chunked_index = next(
                            (i for i, l in enumerate(encoder.layers)
                             if hasattr(l, 'chunk_size') and getattr(l, 'sliding_window_latents', True) is None), None
                        )
                        if first_chunked_index is not None:
                            first_chunked = encoder.layers[first_chunked_index]
                            stride = getattr(first_chunked, 'stride', None)
                            if stride is None and hasattr(encoder, 'strides') and len(encoder.strides) > first_chunked_index:
                                stride = encoder.strides[first_chunked_index]
                            if stride and stride > 0:
                                latent_align = max(1, first_chunked.chunk_size // stride)
                align = ds_ratio * latent_align  # For chunked attention, align to chunk size after downsampling
                # Round up to nearest multiple
                target_audio_samples = ((target_audio_samples + align - 1) // align) * align
            
            # Cap at original sample_size (don't exceed what was requested)
            audio_sample_size = min(target_audio_samples, sample_size)

    # Use the (potentially adapted) audio_sample_size for latent size calculation
    latent_sample_size = audio_sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        latent_sample_size = audio_sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, latent_sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    diff_objective = model.diffusion_objective

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation or inversion). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        if inversion_params is not None:
            # we're doing an inversion task
            assert diff_objective in diff_objective in ["rectified_flow"], "inversion is supported in RF models only"
            # RF-Inversion
            inversion_noise = noise # this is not the inverted latent, this is gamma-parameterized noise we use during inversion to guide it in-distribution 
            # Modify the conditioning for inversion
            inversion_conditioning = copy.deepcopy(conditioning)
            inversion_conditioning_tensors = model.conditioner(inversion_conditioning, device)
            inversion_conditioning_inputs = model.get_conditioning_inputs(inversion_conditioning_tensors)
            inversion_conditioning_inputs =  {k: v.type(model_dtype) if v is not None else v for k, v in inversion_conditioning_inputs.items()}
            if inversion_params["inversion_unconditional"]:
                # Unconditional seems better for prompt re-stylization
                cfg_dropout_prob = 1.0
            else:
                # "" prompt w/ cfg=1 seems better for reconstruction
                cfg_dropout_prob = 0.0
                for x in inversion_conditioning:
                    if "prompt" in x: x["prompt"] = ""
            # invert the audio
            inverted_latents = invert_audio(model.model, init_audio, noise=inversion_noise, \
                inversion_params=inversion_params,
                cfg_dropout_prob=cfg_dropout_prob,
                **inversion_conditioning_inputs)
            noise = inverted_latents # from here on, use this as the seed noise. it represents the inverted audio
            init_audio = None

    # Merge positive and negative conditioning inputs
    cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}

    # Extract sampler_type from sampler_kwargs if present
    sampler_type = sampler_kwargs.pop("sampler_type", "euler")

    sampled = sample_diffusion(
        model=model.model,
        noise=noise,
        cond_inputs=cond_inputs,
        diffusion_objective=diff_objective,
        steps=steps,
        cfg_scale=cfg_scale,
        # Varlen support
        conditioning=conditioning,
        sample_rate=model.sample_rate,
        pretransform=model.pretransform,
        mask_padding_attention=mask_padding_attention,
        use_effective_length_for_schedule=use_effective_length_for_schedule,
        headroom_seconds=duration_padding_sec,
        # Timestep schedule
        dist_shift=dist_shift if dist_shift is not None else model.sampling_dist_shift,
        # Sampler options
        sampler_type=sampler_type,
        batch_cfg=True,
        rescale_cfg=True,
        apg_scale=apg_scale,
        # Init data
        init_data=init_audio,
        init_noise_level=init_noise_level,
        # Other
        decode=not return_latents,
        **sampler_kwargs
    )

    return sampled

def generate_diffusion_cond_inpaint(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask = None,
        inpaint_mask_start_seconds: tp.Optional[float] = None,
        inpaint_mask_end_seconds: tp.Optional[float] = None,
        return_latents = False,
        adapt_duration_to_conditioning: tp.Optional[bool] = None,
        duration_padding_sec: float = 6.0,
        use_effective_length_for_schedule: tp.Optional[bool] = None,
        mask_padding_attention: tp.Optional[bool] = None,
        apg_scale: float = 1.0,
        dist_shift = None,
        **sampler_kwargs
        ) -> torch.Tensor:
    """
    Generate audio from a prompt using a diffusion inpainting model.

    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        inpaint_mask: A prebuilt mask tensor for inpainting. Shape should be [batch_size, sample_size].
            When adapt_duration_to_conditioning is True, the mask length should match the adapted audio_sample_size.
            Ignored if inpaint_mask_start_seconds/inpaint_mask_end_seconds are provided.
        inpaint_mask_start_seconds: Start of the inpaint region in seconds. The mask will be built internally
            at the correct audio_sample_size (after duration adaptation), so positions are always accurate.
        inpaint_mask_end_seconds: End of the inpaint region in seconds.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        adapt_duration_to_conditioning: Adapt sample size based on seconds_total + duration_padding_sec.
            None (default) = auto-detect from model; True/False = explicit override.
        duration_padding_sec: Extra seconds to add when adapting duration (default 6.0).
        use_effective_length_for_schedule: Use effective_seq_len for distribution shift.
            None (default) = auto-detect from model; True/False = explicit override.
        mask_padding_attention: Create padding_mask for attention based on effective_seq_len.
            None (default) = auto-detect from model; True/False = explicit override.
        apg_scale: APG (Adaptive Projected Guidance) scale. 1.0 = full APG, 0.0 = vanilla CFG.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.
    """

    if mask_padding_attention is None:
        mask_padding_attention = getattr(model, 'mask_padding_attention', False)
    if use_effective_length_for_schedule is None:
        use_effective_length_for_schedule = getattr(model, 'use_effective_length_for_schedule', False)
    if adapt_duration_to_conditioning is None:
        adapt_duration_to_conditioning = mask_padding_attention or use_effective_length_for_schedule

    # The length of the output in audio samples
    audio_sample_size = sample_size

    # Optionally adapt sample size based on seconds_total conditioning
    if adapt_duration_to_conditioning and conditioning is not None:
        # Find the maximum seconds_total across the batch
        max_seconds = 0.0
        for cond_dict in conditioning:
            if "seconds_total" in cond_dict:
                max_seconds = max(max_seconds, cond_dict["seconds_total"])
        
        if max_seconds > 0:
            # Calculate target audio samples with padding, capped at original sample_size
            target_audio_samples = int((max_seconds + duration_padding_sec) * model.sample_rate)

            # Ensure we align to pretransform downsampling ratio if applicable
            if model.pretransform is not None:
                # Round up to nearest multiple of downsampling ratio, and also align to chunk size if using chunked attention
                ds_ratio = model.pretransform.downsampling_ratio
                latent_align = 1
                # Only perform if the encoder has chunked attention layers and not sliding window 
                if (hasattr(model.pretransform, 'model') and
                        hasattr(model.pretransform.model, 'encoder')):
                    encoder = model.pretransform.model.encoder
                    if hasattr(encoder, 'layers'):
                        first_chunked_index = next(
                            (i for i, l in enumerate(encoder.layers)
                             if hasattr(l, 'chunk_size') and getattr(l, 'sliding_window_latents', True) is None), None
                        )
                        if first_chunked_index is not None:
                            first_chunked = encoder.layers[first_chunked_index]
                            stride = getattr(first_chunked, 'stride', None)
                            if stride is None and hasattr(encoder, 'strides') and len(encoder.strides) > first_chunked_index:
                                stride = encoder.strides[first_chunked_index]
                            if stride and stride > 0:
                                latent_align = max(1, first_chunked.chunk_size // stride)
                align = ds_ratio * latent_align  # For chunked attention, align to chunk size after downsampling
                # Round up to nearest multiple
                target_audio_samples = ((target_audio_samples + align - 1) // align) * align
            
            # Cap at original sample_size (don't exceed what was requested)
            audio_sample_size = min(target_audio_samples, sample_size)

    # Use the (potentially adapted) audio_sample_size for latent size calculation
    latent_sample_size = audio_sample_size

    # If this is latent diffusion, change sample_size to the downsampled latent size
    if model.pretransform is not None:
        latent_sample_size = audio_sample_size // model.pretransform.downsampling_ratio
    
    # Keep sample_size variable pointing to latent size for backward compatibility in rest of function
    sample_size = latent_sample_size
    
    # Build inpaint mask from seconds if provided, using the (potentially adapted) audio_sample_size.
    # This ensures mask positions are correct regardless of duration adaptation.
    if inpaint_mask_start_seconds is not None and inpaint_mask_end_seconds is not None:
        mask_start_samples = int(inpaint_mask_start_seconds * model.sample_rate)
        mask_end_samples = int(inpaint_mask_end_seconds * model.sample_rate)
        # Clamp to audio_sample_size
        mask_start_samples = min(mask_start_samples, audio_sample_size)
        mask_end_samples = min(mask_end_samples, audio_sample_size)
        inpaint_mask = torch.ones(1, audio_sample_size, device=device)
        inpaint_mask[:, mask_start_samples:mask_end_samples] = 0

    # If the caller passed a prebuilt mask sized to the un-adapted sample_size (or
    # anything longer than audio_sample_size), truncate to audio_sample_size so the
    # downstream nearest-neighbor interpolation preserves the mask's time-domain
    # positions instead of squashing the mask region.
    if inpaint_mask is not None and inpaint_mask.shape[-1] > audio_sample_size:
        inpaint_mask = inpaint_mask[:, :audio_sample_size]

    # Match training: when mask_padding_attention is used, random_inpaint_mask
    # zeroes the mask past real_sequence_length (models/inpainting.py). Apply the
    # same convention here so the mask matches the training distribution, whether
    # it was built from seconds above or passed in by the caller.
    if inpaint_mask is not None and mask_padding_attention and conditioning is not None:
        max_seconds = max(
            (c.get("seconds_total", 0.0) for c in conditioning), default=0.0
        )
        if max_seconds > 0:
            effective_audio_len = int(max_seconds * model.sample_rate)
            mask_len = inpaint_mask.shape[-1]
            if effective_audio_len < mask_len:
                inpaint_mask = inpaint_mask.clone()
                inpaint_mask[:, effective_audio_len:] = 0

    if inpaint_mask is not None:
        inpaint_mask = inpaint_mask.float()

    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=init_audio.shape[-1], mode='nearest').squeeze(1)

        init_audio = init_audio.repeat(batch_size, 1, 1)

    if inpaint_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        inpaint_sr, inpaint_audio = inpaint_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        inpaint_audio = prepare_audio(inpaint_audio, in_sr=inpaint_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            inpaint_audio = model.pretransform.encode(inpaint_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=inpaint_audio.shape[-1], mode='nearest').squeeze(1)

        inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
    else:
       
        if inpaint_mask is not None:
            # interpolate inpaint mask to the sample size
            inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=sample_size, mode='nearest').squeeze(1)

    if inpaint_mask is None:
        mask = torch.zeros((batch_size, 1, sample_size), device=device)  
    else:
        mask = inpaint_mask.unsqueeze(1)

    # Inpainting mask
    mask = mask.to(device)

    if inpaint_audio is not None:
        inpaint_input = inpaint_audio * mask.expand_as(inpaint_audio)
    else:
        inpaint_input = torch.zeros((batch_size, model.io_channels, sample_size), device=device)

    conditioning_tensors['inpaint_mask'] = [mask]
    conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors['inpaint_mask'] = [mask]
        negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    
    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    # Merge positive and negative conditioning inputs
    cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}

    # Extract sampler_type from sampler_kwargs if present
    sampler_type = sampler_kwargs.pop("sampler_type", "euler")

    sampled = sample_diffusion(
        model=model.model,
        noise=noise,
        cond_inputs=cond_inputs,
        diffusion_objective=model.diffusion_objective,
        steps=steps,
        cfg_scale=cfg_scale,
        # Varlen support
        conditioning=conditioning,
        sample_rate=model.sample_rate,
        pretransform=model.pretransform,
        mask_padding_attention=mask_padding_attention,
        use_effective_length_for_schedule=use_effective_length_for_schedule,
        headroom_seconds=duration_padding_sec,
        # Timestep schedule
        dist_shift=dist_shift if dist_shift is not None else model.sampling_dist_shift,
        # Sampler options
        sampler_type=sampler_type,
        batch_cfg=True,
        rescale_cfg=True,
        apg_scale=apg_scale,
        # Init data
        init_data=init_audio,
        init_noise_level=init_noise_level,
        # Other
        decode=not return_latents,
        **sampler_kwargs
    )

    return sampled


# builds a softmask given the parameters
# returns array of values 0 to 1, size sample_size, where 0 means noise / fresh generation, 1 means keep the input audio, 
# and anything between is a mixture of old/new
# ideally 0.5 is half/half mixture but i haven't figured this out yet
def build_mask(sample_size, mask_args):
    maskstart = math.floor(mask_args["maskstart"]/100.0 * sample_size)
    maskend = math.ceil(mask_args["maskend"]/100.0 * sample_size)
    softnessL = round(mask_args["softnessL"]/100.0 * sample_size)
    softnessR = round(mask_args["softnessR"]/100.0 * sample_size)
    marination = mask_args["marination"]
    # use hann windows for softening the transition (i don't know if this is correct)
    hannL = torch.hann_window(softnessL*2, periodic=False)[:softnessL]
    hannR = torch.hann_window(softnessR*2, periodic=False)[softnessR:]
    # build the mask. 
    mask = torch.zeros((sample_size))
    mask[maskstart:maskend] = 1
    mask[maskstart:maskstart+softnessL] = hannL
    mask[maskend-softnessR:maskend] = hannR
    # marination finishes the inpainting early in the denoising schedule, and lets audio get changed in the final rounds
    if marination > 0:        
        mask = mask * (1-marination) 
    #print(mask)
    return mask
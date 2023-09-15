import numpy as np
import torch 
import typing as tp

from torchaudio import transforms as T

from .sampling import sample_k
from .utils import prepare_audio

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
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)

    torch.manual_seed(seed)

    if init_audio is not None:
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level

        noise = torch.randn_like(init_audio)

        sampled = sample_k(model.model, noise, steps=steps, init_data=init_audio, **sampler_kwargs, device=device)
    else:
        if model.pretransform is not None:
            sample_size = sample_size // model.pretransform.downsampling_ratio

        noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)
        sampled = sample_k(model.model, noise, steps, **sampler_kwargs, device=device)

    if model.pretransform is not None and not return_latents:
        sampled = model.pretransform.decode(sampled)

    return sampled


def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
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
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)

    torch.manual_seed(seed)

    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"

    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    
    conditioning_tensors = model.get_conditioning_inputs(conditioning_tensors)

    if init_audio is not None:
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level

        noise = torch.randn_like(init_audio)

        sampled = sample_k(model.model, noise, steps=steps, init_data=init_audio, **sampler_kwargs, **conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, scale_cfg=True, device=device)
    else:
        if model.pretransform is not None:
            sample_size = sample_size // model.pretransform.downsampling_ratio

        noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)
        sampled = sample_k(model.model, noise, steps, **sampler_kwargs, **conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, scale_cfg=True, device=device)

    if model.pretransform is not None and not return_latents:
        sampled = model.pretransform.decode(sampled)

    return sampled


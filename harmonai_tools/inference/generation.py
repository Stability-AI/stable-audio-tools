import numpy as np
import torch 
import typing as tp

from torchaudio import transforms as T

from einops import rearrange

#from ..models.diffusion import ConditionedDiffusionModelWrapper
from .sampling import sample, sample_k, variation_k
from ..data.utils import PadCrop


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
        **sampler_kwargs
        ) -> torch.Tensor: 
    """Generate audio from a prompt using a diffusion model."""

    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)

    print(seed)

    torch.manual_seed(seed)

    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"

    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    
    conditioning_tensors = model.get_conditioning_inputs(conditioning_tensors)

    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    
    if init_audio is not None:
        in_sr, init_audio = init_audio

        init_audio = init_audio.to(device)

        print(f"Init audio: {init_audio}")

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(device)
            init_audio = resample_tf(init_audio)

        init_audio = PadCrop(sample_size, randomize=False)(init_audio)

        # Add batch dimension if needed
        if len(init_audio.shape) == 2:
            init_audio = init_audio.unsqueeze(0)       

        io_channels = model.io_channels

        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        if io_channels == 1:
            # Convert to mono
            init_audio = init_audio.mean(1, keepdim=True)

        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        sampled = variation_k(model.model, init_audio, init_noise_level, steps, **sampler_kwargs, **conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, scale_cfg=True)
    else:
        if model.pretransform is not None:
            sample_size = sample_size // model.pretransform.downsampling_ratio

        noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)
        sampled = sample_k(model.model, noise, steps, **sampler_kwargs, **conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, scale_cfg=True)


    if model.pretransform is not None:
        sampled = model.pretransform.decode(sampled)

    return sampled


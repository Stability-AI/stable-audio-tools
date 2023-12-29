from .generation import generate_diffusion_cond
from ..models.diffusion import ConditionedDiffusionModelWrapper
from ..inference.utils import prepare_audio

import torch
from torchaudio import transforms as T
from torch.nn import functional as F

def generate_mono_to_stereo(
        model: ConditionedDiffusionModelWrapper, 
        audio: torch.Tensor, # (batch, channels, time)
        in_sr: int,
        steps: int,
        sampler_kwargs: dict = {},
    ):
    """
    Generate stereo audio from mono audio using a diffusion model.

    Args:
        model: A mono-to-stereo diffusion prior wrapper
        audio: The mono audio to convert to stereo
        in_sr: The sample rate of the input audio
        steps: The number of diffusion steps to run
        sampler_kwargs: Keyword arguments to pass to the diffusion sampler
    """

    device = audio.device

    sample_rate = model.sample_rate

    # Resample input audio if necessary
    if in_sr != sample_rate:
        resample_tf = T.Resample(in_sr, sample_rate).to(audio.device)
        audio = resample_tf(audio)

    audio_length = audio.shape[-1]

    # Pad input audio to be compatible with the model
    min_length = model.min_input_length
    padded_input_length = audio_length + (min_length - (audio_length % min_length)) % min_length

    # Pad input audio to be compatible with the model
    if padded_input_length > audio_length:
        audio = F.pad(audio, (0, padded_input_length - audio_length))

    # Make audio mono, duplicate to stereo
    dual_mono = audio.mean(1, keepdim=True).repeat(1, 2, 1)

    if model.pretransform is not None:
        dual_mono = model.pretransform.encode(dual_mono)

    conditioning = {"source": [dual_mono]}

    stereo_audio = generate_diffusion_cond(
        model, 
        conditioning_tensors=conditioning,
        steps=steps,
        sample_size=padded_input_length,
        sample_rate=sample_rate,
        device=device,
        **sampler_kwargs,
    ) 

    return stereo_audio


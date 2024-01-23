from enum import Enum
import typing as tp

from .diffusion import ConditionedDiffusionModelWrapper
from ..inference.generation import generate_diffusion_cond
from ..inference.utils import prepare_audio

import torch
from torch.nn import functional as F
from torchaudio import transforms as T

# Define prior types enum
class PriorType(Enum):
    MonoToStereo = 1
    SourceSeparation = 2

class DiffusionPrior(ConditionedDiffusionModelWrapper):
    def __init__(self, *args, prior_type: PriorType=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_type = prior_type  

class MonoToStereoDiffusionPrior(DiffusionPrior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, prior_type=PriorType.MonoToStereo, **kwargs)

    def stereoize(
        self, 
        audio: torch.Tensor, # (batch, channels, time)
        in_sr: int,
        steps: int,
        sampler_kwargs: dict = {},
    ):
        """
        Generate stereo audio from mono audio using a pre-trained diffusion prior

        Args:
            audio: The mono audio to convert to stereo
            in_sr: The sample rate of the input audio
            steps: The number of diffusion steps to run
            sampler_kwargs: Keyword arguments to pass to the diffusion sampler
        """

        device = audio.device

        sample_rate = self.sample_rate

        # Resample input audio if necessary
        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(audio.device)
            audio = resample_tf(audio)

        audio_length = audio.shape[-1]

        # Pad input audio to be compatible with the model
        min_length = self.min_input_length
        padded_input_length = audio_length + (min_length - (audio_length % min_length)) % min_length

        # Pad input audio to be compatible with the model
        if padded_input_length > audio_length:
            audio = F.pad(audio, (0, padded_input_length - audio_length))

        # Make audio mono, duplicate to stereo
        dual_mono = audio.mean(1, keepdim=True).repeat(1, 2, 1)

        if self.pretransform is not None:
            dual_mono = self.pretransform.encode(dual_mono)

        conditioning = {"source": [dual_mono]}

        stereo_audio = generate_diffusion_cond(
            self, 
            conditioning_tensors=conditioning,
            steps=steps,
            sample_size=padded_input_length,
            sample_rate=sample_rate,
            device=device,
            **sampler_kwargs,
        ) 

        return stereo_audio


class SourceSeparationDiffusionPrior(DiffusionPrior):
    """
    A diffusion prior model made for conditioned source separation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, prior_type=PriorType.SourceSeparation, **kwargs)

    def separate(
        self, 
        mixed_audio: torch.Tensor, # (batch, channels, time)
        in_sr: int,
        steps: int,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        sampler_kwargs: dict = {},
    ):
        """
        Separate audio sources based on conditioning using a pre-trained diffusion prior

        Args:
            mixed_audio: The mixed audio to separate
            in_sr: The sample rate of the input audio
            steps: The number of diffusion steps to run
            conditioning: The conditioning to use for source separation
            conditioning_tensors: Pre-computed conditioning tensors to use for source separation. If provided, conditioning is ignored.
            sampler_kwargs: Keyword arguments to pass to the diffusion sampler
        """

        device = mixed_audio.device

        sample_rate = self.sample_rate

        # Resample input audio if necessary
        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(mixed_audio.device)
            mixed_audio = resample_tf(mixed_audio)

        audio_length = mixed_audio.shape[-1]

        # Pad input audio to be compatible with the model
        min_length = self.min_input_length
        padded_input_length = audio_length + (min_length - (audio_length % min_length)) % min_length

        # Pad input audio to be compatible with the model
        if padded_input_length > audio_length:
            mixed_audio = F.pad(mixed_audio, (0, padded_input_length - audio_length))

        if self.pretransform is not None:
            mixed_audio = self.pretransform.encode(mixed_audio)

        # Conditioning
        assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors for conditioned source separation"
        if conditioning_tensors is None:
            conditioning_tensors = self.conditioner(conditioning, device)
        
        # Pass in the mixture audio as conditioning
        conditioning_tensors["source"] = [mixed_audio]

        stereo_audio = generate_diffusion_cond(
            self, 
            conditioning_tensors=conditioning_tensors,
            steps=steps,
            sample_size=padded_input_length,
            sample_rate=sample_rate,
            device=device,
            **sampler_kwargs,
        ) 

        return stereo_audio
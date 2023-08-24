import numpy as np
import torch 
import typing as tp
import math 
from torchaudio import transforms as T

from einops import rearrange

#from ..models.diffusion import ConditionedDiffusionModelWrapper
from .sampling import sample, sample_k
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
        mask_args: dict = None,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """Generate audio from a prompt using a diffusion model."""

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_tensors = model.get_conditioning_inputs(conditioning_tensors)

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio
        init_audio = init_audio.to(device)
        # Resample the raw audio to our model's sample rate
        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(device)
            init_audio = resample_tf(init_audio)

        # If the init audio is shorter than our model's window, pad it with zeros
        init_audio = PadCrop(audio_sample_size, randomize=False)(init_audio)
        # Add batch dimension if needed
        if len(init_audio.shape) == 2:
            init_audio = init_audio.unsqueeze(0)
        elif len(init_audio.shape) == 1:
            init_audio = init_audio.unsqueeze(0).unsqueeze(0)

        # Adjust for mono or stereo models
        io_channels = model.io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels
        if io_channels == 1:
            # Convert to mono
            init_audio = init_audio.mean(1, keepdim=True)
        elif io_channels == 2:
            # Convert to stereo
            if init_audio.shape[1] == 1:
                init_audio = init_audio.repeat(1, 2, 1)
            elif init_audio.shape[1] > 2:
                init_audio = init_audio[:, :2, :]
        # Finally, if this is latent diffusion, run the raw audio through the VAE encoder
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
        # Okay great, our input audio has been prepared. 
    else:
        # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch. 
        init_audio = None
        init_noise_level = None
        mask_args = None

    # Inpainting mask
    if init_audio is not None and mask_args is not None:
        # Cut and paste init_audio according to cropfrom, pastefrom, pasteto
        # This is helpful for forward and reverse outpainting
        cropfrom = math.floor(mask_args["cropfrom"]/100.0 * sample_size)
        pastefrom = math.floor(mask_args["pastefrom"]/100.0 * sample_size)
        pasteto = math.ceil(mask_args["pasteto"]/100.0 * sample_size)
        if pastefrom > pasteto: 
            # if from/to are out of order, swap them
            pasteto = _
            pastefrom = pasteto 
            pasteto = _ 
        croplen = pasteto - pastefrom
        if cropfrom + croplen > sample_size:
            croplen = sample_size - cropfrom 
        cropto = cropfrom + croplen
        pasteto = pastefrom + croplen
        cutpaste = init_audio.new_zeros(init_audio.shape)
        cutpaste[:, :, pastefrom:pasteto] = init_audio[:,:,cropfrom:cropto]
        #print(cropfrom, cropto, pastefrom, pasteto)
        init_audio = cutpaste
        # Build a soft mask (list of floats 0 to 1, the size of the latent) from the given args
        mask = build_mask(sample_size, mask_args)
        mask = mask.to(device)
    else:
        mask = None

    # Now the generative AI part:
    # k-diffusion denoising process go!
    sampled = sample_k(model.model, noise, init_audio, init_noise_level, mask, steps, **sampler_kwargs, **conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, scale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)

    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None:
        sampled = model.pretransform.decode(sampled)

    # Return audio
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
    marination = math.floor(mask_args["marination"]/100.0)
    # use hann windows for softening the transition (i don't know if this is correct)
    hannL = torch.hann_window(softnessL*2, periodic=False)[:softnessL]
    hannR = torch.hann_window(softnessR*2, periodic=False)[softnessR:]
    # build the mask. 
    mask = torch.zeros((sample_size))
    mask[maskstart:maskend] = 1
    mask[maskstart:maskstart+softnessL] = hannL
    mask[maskend-softnessR:maskend] = hannR
    # marination finishes the inpainting early in the denoising schedule, and lets audio get changed in the final round
    if marination > 0:        
        mask = mask * (1-marination)
    #print(mask)
    return mask
import torch
import math
from tqdm import trange

import k_diffusion as K

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(model, x, steps, eta, **extra_args):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred



def sample_k(model_fn, noise, steps=100, sampler_type="dpmpp-2m-sde", sigma_min=0.5, sigma_max=50, rho=1.0, device="cuda", **extra_args):

    denoiser = K.external.VDenoiser(model_fn)
    sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)

    noise = noise * sigmas[0]

    with torch.cuda.amp.autocast():
        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, noise, sigmas, disable=False, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, noise, sigmas, disable=False, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, noise, sigmas, disable=False, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, noise, sigmas, disable=False, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, noise, sigma_min, sigma_max, steps, disable=False, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, noise, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, noise, sigmas, disable=False, extra_args=extra_args)
        

def variation_k(model_fn, init_audio, init_noise_level=0.1, steps=100, sampler_type="dpmpp-2m-sde", sigma_max=80, sigma_min=0.5, rho=1.0, device="cuda", **extra_args):

    denoiser = K.external.VDenoiser(model_fn)
    sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, init_noise_level, rho, device=device)

    init_audio = init_audio + torch.randn_like(init_audio) * init_noise_level #sigmas[0]

    print(f"sigmas[0]: {sigmas[0]}")

    with torch.cuda.amp.autocast():
        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, init_audio, sigmas, disable=False, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, init_audio, sigmas, disable=False, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, init_audio, sigmas, disable=False, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, init_audio, sigmas, disable=False, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, init_audio, sigma_min, init_noise_level, steps, disable=False, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, init_audio, sigma_min, init_noise_level, rtol=0.01, atol=0.01, disable=False, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, init_audio, sigmas, disable=False, extra_args=extra_args)
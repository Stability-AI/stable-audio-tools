import torch
import math
from tqdm import trange, tqdm

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

def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@torch.no_grad()
def sample_discrete_euler(model, x, steps, sigma_max=1, **extra_args):
    """Draws samples from a model given starting noise. Euler method"""

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)

    #alphas, sigmas = 1-t, t

    for t_curr, t_prev in tqdm(zip(t[:-1], t[1:])):
            # Broadcast the current timestep to the correct shape
            t_curr_tensor = t_curr * torch.ones(
                (x.shape[0],), dtype=x.dtype, device=x.device
            )
            dt = t_prev - t_curr  # we solve backwards in our formulation
            x = x + dt * model(x, t_curr_tensor, **extra_args) #.denoise(x, denoiser, t_curr_tensor, cond, uc)

    # If we are on the last timestep, output the denoised image
    return x

@torch.no_grad()
def sample(model, x, steps, eta, **extra_args):
    """Draws samples from a model given starting noise. v-diffusion"""
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

# Soft mask inpainting is just shrinking hard (binary) mask inpainting
# Given a float-valued soft mask (values between 0 and 1), get the binary mask for this particular step
def get_bmask(i, steps, mask):
    strength = (i+1)/(steps)
    # convert to binary mask
    bmask = torch.where(mask<=strength,1,0)
    return bmask

def make_cond_model_fn(model, cond_fn):
    def cond_model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return cond_model_fn

# Uses k-diffusion from https://github.com/crowsonkb/k-diffusion
# init_data is init_audio as latents (if this is latent diffusion)
# For sampling, set both init_data and mask to None
# For variations, set init_data 
# For inpainting, set both init_data & mask 
def sample_k(
        model_fn, 
        noise, 
        init_data=None,
        mask=None,
        steps=100, 
        sampler_type="dpmpp-2m-sde", 
        sigma_min=0.5, 
        sigma_max=50, 
        rho=1.0, device="cuda", 
        callback=None, 
        cond_fn=None,
        **extra_args
    ):

    denoiser = K.external.VDenoiser(model_fn)

    if cond_fn is not None:
        denoiser = make_cond_model_fn(denoiser, cond_fn)

    # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
    sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
    # Scale the initial noise by sigma 
    noise = noise * sigmas[0]

    wrapped_callback = callback

    if mask is None and init_data is not None:
        # VARIATION (no inpainting)
        # set the initial latent to the init_data, and noise it with initial sigma
        x = init_data + noise 
    elif mask is not None and init_data is not None:
        # INPAINTING
        bmask = get_bmask(0, steps, mask)
        # initial noising
        input_noised = init_data + noise
        # set the initial latent to a mix of init_data and noise, based on step 0's binary mask
        x = input_noised * bmask + noise * (1-bmask)
        # define the inpainting callback function (Note: side effects, it mutates x)
        # See https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py#L596C13-L596C105
        # callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        # This is called immediately after `denoised = model(x, sigmas[i] * s_in, **extra_args)`
        def inpainting_callback(args):
            i = args["i"]
            x = args["x"]
            sigma = args["sigma"]
            #denoised = args["denoised"]
            # noise the init_data input with this step's appropriate amount of noise
            input_noised = init_data + torch.randn_like(init_data) * sigma
            # shrinking hard mask
            bmask = get_bmask(i, steps, mask)
            # mix input_noise with x, using binary mask
            new_x = input_noised * bmask + x * (1-bmask)
            # mutate x
            x[:,:,:] = new_x[:,:,:]
        # wrap together the inpainting callback and the user-submitted callback. 
        if callback is None: 
            wrapped_callback = inpainting_callback
        else:
            wrapped_callback = lambda args: (inpainting_callback(args), callback(args))
    else:
        # SAMPLING
        # set the initial latent to noise
        x = noise


    with torch.cuda.amp.autocast():
        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-3m-sde":
            return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)

# Uses discrete Euler sampling for rectified flow models
# init_data is init_audio as latents (if this is latent diffusion)
# For sampling, set both init_data and mask to None
# For variations, set init_data 
# For inpainting, set both init_data & mask 
def sample_rf(
        model_fn, 
        noise, 
        init_data=None,
        steps=100, 
        sigma_max=1,
        device="cuda", 
        callback=None, 
        cond_fn=None,
        **extra_args
    ):

    if sigma_max > 1:
        sigma_max = 1

    if cond_fn is not None:
        denoiser = make_cond_model_fn(denoiser, cond_fn)

    wrapped_callback = callback

    if init_data is not None:
        # VARIATION (no inpainting)
        # Interpolate the init data and the noise for init audio
        x = init_data * (1 - sigma_max) + noise * sigma_max
    else:
        # SAMPLING
        # set the initial latent to noise
        x = noise

    with torch.cuda.amp.autocast():
        # TODO: Add callback support
        #return sample_discrete_euler(model_fn, x, steps, sigma_max, callback=wrapped_callback, **extra_args)
        return sample_discrete_euler(model_fn, x, steps, sigma_max, **extra_args)
import torch
import math
from tqdm import trange, tqdm
import torch.distributions as dist

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

class DistributionShift:
    def __init__(self, base_shift=0.5, max_shift=1.15, max_length=4096, min_length=256, use_sine=False):
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.max_length = max_length
        self.min_length = min_length
        self.use_sine = use_sine
    
    def time_shift(self, t: torch.Tensor, seq_len: int):
        sigma = 1.0
        mu = - (self.base_shift + (self.max_shift - self.base_shift) * (seq_len - self.min_length) / (self.max_length - self.min_length))
        t_out = 1 - math.exp(mu) / (math.exp(mu) + (1 / (1 - t) - 1) ** sigma)

        if self.use_sine:
            t_out = torch.sin(t_out * math.pi / 2)

        return t_out

def truncated_logistic_normal_rescaled(shape, left_trunc=0.075, right_trunc=1):
    """
  
    shape: shape of the output tensor
    left_trunc: left truncation point, fraction of probability to be discarded
    right_trunc: right truncation boundary, should be 1 (never seen at test time)
    """
    
    # Step 1: Sample from the logistic normal distribution (sigmoid of normal)
    logits = torch.randn(shape)
    
    # Step 2: Apply the CDF transformation of the normal distribution
    normal_dist = dist.Normal(0, 1)
    cdf_values = normal_dist.cdf(logits)
    
    # Step 3: Define the truncation bounds on the CDF
    lower_bound = normal_dist.cdf(torch.logit(torch.tensor(left_trunc)))
    upper_bound = normal_dist.cdf(torch.logit(torch.tensor(right_trunc)))

    # Step 4: Rescale linear CDF values into the truncated region (between lower_bound and upper_bound)
    truncated_cdf_values = lower_bound + (upper_bound - lower_bound) * cdf_values
    
    # Step 5: Map back to logistic-normal space using inverse CDF
    truncated_samples = torch.sigmoid(normal_dist.icdf(truncated_cdf_values))
    
    # Step 6: Rescale values so that min is 0 and max is just below 1
    rescaled_samples = (truncated_samples - left_trunc) / (right_trunc - left_trunc)

    return rescaled_samples

@torch.no_grad()
def sample_discrete_euler(model, x, steps, sigma_max=1, callback=None, dist_shift=None, **extra_args):
    """Draws samples from a model given starting noise. Euler method"""

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)

    if dist_shift is not None:
        t = dist_shift.time_shift(t, x.shape[-1])

    #alphas, sigmas = 1-t, t

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]))):
        # Broadcast the current timestep to the correct shape
        t_curr_tensor = t_curr * torch.ones(
            (x.shape[0],), dtype=x.dtype, device=x.device
        )
        dt = t_prev - t_curr  # we solve backwards in our formulation
        v = model(x, t_curr_tensor, **extra_args)
        x = x + dt * v

        if callback is not None:
            denoised = x - t_prev * v
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i+1, 'denoised': denoised })

    # If we are on the last timestep, output the denoised data
    return x

@torch.no_grad()
def sample_rk4(model, x, steps, sigma_max=1, callback=None, dist_shift=None, **extra_args):
    """Draws samples from a model given starting noise. 4th-order Runge-Kutta"""

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)

    if dist_shift is not None:
        t = dist_shift.time_shift(t, x.shape[-1])

    #alphas, sigmas = 1-t, t

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]))):
        # Broadcast the current timestep to the correct shape
        t_curr_tensor = t_curr * ts
        dt = t_prev - t_curr  # we solve backwards in our formulation
        
        k1 = model(x, t_curr_tensor, **extra_args)
        k2 = model(x + dt / 2 * k1, (t_curr + dt / 2) * ts, **extra_args)
        k3 = model(x + dt / 2 * k2, (t_curr + dt / 2) * ts, **extra_args)
        k4 = model(x + dt * k3, t_prev * ts, **extra_args)
        
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        if callback is not None:
            denoised = x - t_prev * k4
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i+1, 'denoised': denoised })

    # If we are on the last timestep, output the denoised data
    return x

@torch.no_grad()
def sample_flow_dpmpp(model, x, steps, sigma_max=1, callback=None, dist_shift=None, **extra_args):
    """Draws samples from a model given starting noise. DPM-Solver++ for RF models"""

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)

    if dist_shift is not None:
        t = dist_shift.time_shift(t, x.shape[-1])

    old_denoised = None

    log_snr = lambda t: ((1-t) / t).log()

    for i in trange(len(t) - 1, disable=False):
        denoised = x - t[i] * model(x, t[i] * ts, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': t[i], 'sigma_hat': t[i], 'denoised': denoised})
        t_curr, t_next = t[i], t[i + 1]
        alpha_t = 1-t_next
        h = log_snr(t_next) - log_snr(t_curr)
        if old_denoised is None or t_next == 0:
            x = (t_next / t_curr) * x - alpha_t * (-h).expm1() * denoised
        else:
            h_last = log_snr(t_curr) - log_snr(t[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_next / t_curr) * x - alpha_t * (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


@torch.no_grad()
def sample(model, x, steps, eta, callback=None, sigma_max=1.0, dist_shift=None, cfg_pp=False, **extra_args):
    """Draws samples from a model given starting noise. v-diffusion"""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)[:-1]

    if dist_shift is not None:
        t = dist_shift.time_shift(t, x.shape[-1])

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        if cfg_pp:
            # Get the model output (v, the predicted velocity)
            v, info = model(x, ts * t[i], return_info=True, **extra_args)

            if "uncond_output" in info:
                v_eps = info["uncond_output"]
            else:
                v_eps = v
        else:
            v = model(x, ts * t[i], **extra_args)
            v_eps = v

        # Predict the noise and the denoised data
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v_eps * alphas[i]

        # If we are not on the last timestep, compute the noisy data for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised data in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

        if callback is not None:
            denoised = pred
            callback({'x': x, 't': t[i], 'sigma': sigmas[i], 'i': i, 'denoised': denoised })

    # If we are on the last timestep, output the denoised data
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
# For sampling, init_data to none
# For variations, set init_data 
def sample_k(
        model_fn, 
        noise, 
        init_data=None,
        steps=100, 
        sampler_type="dpmpp-2m-sde", 
        sigma_min=0.01, 
        sigma_max=100, 
        rho=1.0, 
        device="cuda", 
        callback=None, 
        cond_fn=None,
        **extra_args
    ):

    is_k_diff = sampler_type in ["k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde","dpmpp-2m"]
    is_v_diff = sampler_type in ["v-ddim", "v-ddim-cfgpp"]

    if is_k_diff:

        denoiser = K.external.VDenoiser(model_fn)

        if cond_fn is not None:
            denoiser = make_cond_model_fn(denoiser, cond_fn)

        # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
        sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
        # Scale the initial noise by sigma 
        noise = noise * sigmas[0]

        if init_data is not None:
            # set the initial latent to the init_data, and noise it with initial sigma
            x = init_data + noise 
        else:
            # SAMPLING
            # set the initial latent to noise
            x = noise

        
        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m":
            return K.sampling.sample_dpmpp_2m(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-3m-sde":
            return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
    elif is_v_diff:

        if sigma_max > 1: # sigma_max should be between 0 and 1
            sigma_max = 1

        if cond_fn is not None:
            model_fn = make_cond_model_fn(model_fn, cond_fn)

        alpha, sigma = t_to_alpha_sigma(torch.tensor(sigma_max))

        if init_data is not None:
            x = init_data * alpha + noise * sigma
        else:
            x = noise

        if sampler_type == "v-ddim" or sampler_type == "v-ddim-cfgpp":
            use_cfg_pp = sampler_type == "v-ddim-cfgpp"
            return sample(model_fn, x, steps, eta=0.0, sigma_max=sigma_max, cfg_pp=use_cfg_pp, callback=callback, **extra_args)
    else:
        raise ValueError(f"Unknown sampler type {sampler_type}")

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
        sampler_type="euler",
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

    if init_data is not None:

        if "dist_shift" in extra_args:
            dist_shift = extra_args["dist_shift"]

            # Shift the sigma_max value for init audio to account for the time shift in the sampler
            if sigma_max < 1:
                sigma_max = dist_shift.time_shift(torch.tensor(sigma_max), init_data.shape[-1]).item()

        # VARIATION (no inpainting)
        # Interpolate the init data and the noise for init audio
        x = init_data * (1 - sigma_max) + noise * sigma_max
    else:
        # SAMPLING
        # set the initial latent to noise
        x = noise

    if sampler_type == "euler":
        return sample_discrete_euler(model_fn, x, steps, sigma_max, callback=callback, **extra_args)
    elif sampler_type == "rk4":
        return sample_rk4(model_fn, x, steps, sigma_max, callback=callback, **extra_args)
    elif sampler_type == "dpmpp":
        return sample_flow_dpmpp(model_fn, x, steps, sigma_max, callback=callback, **extra_args)
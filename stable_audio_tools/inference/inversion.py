import torch
import math
from tqdm import trange, tqdm
import torch.distributions as dist

@torch.no_grad()
def invert_audio(
    model,
    waveform_latent,
    noise=None,
    inversion_params: dict = None,
    device="cuda",
    **conditioning_inputs
):
    """
    Perform RF-Inversion on audio to obtain the latent noise.
    Technique based on SEMANTIC IMAGE INVERSION AND EDITING USING RECTIFIED STOCHASTIC DIFFERENTIAL EQUATIONS https://arxiv.org/pdf/2410.10792 
    
    Args:
        model: The stable audio model (DIT model)
        waveform_latent: The audio to invert, shape [batch, channels, sequence_length]
        noise: Noise we add during inversion 
        inversion_params: dict  
            # Note: these params are specific to inversion process; different params can be used during denoising 
            inversion_steps: Number of inversion steps. You can get away with 30 or 50, sometimes 200 is best quality for reproduction. But you may want less for style transfer.
            inversion_gamma: (0-1) Higher means guide it more toward gaussian noise. 0.3 - 0.5 is a good range for reproduction. Set to 0 for style transfer.
            inversion_unconditional: (True, False) True means use cfg_dropout. Note: This is supposed to be the same as cfg=0 but it's not, there's likely a bug somewhere
            inversion_sigma_max: Maximum sigma value used during inversion
            inversion_cfg_scale: CFG value used during inversion.
        device: Device to run inversion on
    Returns:
        inverted_latents: The inverted latents (noise) that can generate this audio
        y_1: Random noise sample used for conditioning
    """

    steps = inversion_params.get("inversion_steps", 100) 
    gamma = inversion_params.get("inversion_gamma", 0.3) 
    sigma_max = inversion_params.get("inversion_sigma_max", 1.0) # set this lower for weirder results
    cfg_scale = inversion_params.get("inversion_cfg_scale", 1.0) # you want cfg=1 during inversion, in most cases

    conditioning_inputs["cfg_scale"] = cfg_scale

    # Initialize with the audio sample
    y_t = waveform_latent.to(device)
    
    # Random noise sample for conditioning
    if noise is not None:
        y_1 = noise
    else: 
        y_1 = torch.randn_like(y_t)
    
    # Create time steps using the same logsnr_linear schedule
    logsnr_max = math.log(((1-sigma_max)/sigma_max) + 1e-6) if sigma_max < 1 else -6
    logsnr = torch.linspace(logsnr_max, 2, steps + 1, device=device)
    t = torch.sigmoid(-logsnr)
    t = t.flip(0)
    t[0] = 0  # Start time
    t[-1] = 1  # End time
    # Note: For inversion, we go from 0 to 1, opposite of sampling
    
    # Forward ODE integration loop
    for i in tqdm(range(steps)):
        t_curr, t_next = t[i], t[i + 1]
        
        # Calculate t_i (as fraction from 0 to 1)
        t_i = t_curr.item()
        
        # Broadcast to batch dimension
        t_curr_tensor = torch.full((y_t.shape[0],), t_i, device=device, dtype=y_t.dtype)
        
        # Get unconditional vector field from model (forward prediction)
        u_t_i = model(y_t, t_curr_tensor, **conditioning_inputs)
        
        # Get conditional vector field (to random noise)
        # This is the drift towards the noise as per the paper
        # Note: The paper divides by (1 - t_i), but here it causes values to explode. Removing improves gamma-noise inversion. 
        # Not sure why. (Possibly there is a bug elsewhere?)
        u_t_i_cond = (y_1 - y_t) #/ (1 - t_i + 1e-6)  
        
        # Controlled vector field - combination of model prediction and drift to noise
        u_hat_t_i = u_t_i + gamma * (u_t_i_cond - u_t_i)
        
        # Calculate step size
        dt = t_next - t_curr
        
        # Update y_t
        y_t = y_t + u_hat_t_i * dt.item()
    
    # Return the inverted latents 
    return y_t
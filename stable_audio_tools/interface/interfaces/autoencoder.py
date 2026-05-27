import gc
import numpy as np
import gradio as gr
import json 
import re
import subprocess
import torch
import torchaudio
import math

from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T
from tqdm import trange

from ...interface.aeiou import audio_spectrogram_image, tokens_spectrogram_image
from ...inference.utils import prepare_audio

model = None
model_type = None
sample_rate = 32000
sample_size = 1920000

def slerp(tensor1: torch.Tensor, tensor2: torch.Tensor, t: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs spherical interpolation with magnitude interpolation and calculates local velocity.
    
    Args:
        tensor1: First tensor of shape [batch, channels, sequence]
        tensor2: Second tensor of shape [batch, channels, sequence]
        t: Interpolation factor between 0 and 1
    
    Returns:
        Tuple of (interpolated tensor, velocity tensor) of the same shape
    """
    # Calculate magnitudes
    mag1 = torch.norm(tensor1, dim=1, keepdim=True)
    mag2 = torch.norm(tensor2, dim=1, keepdim=True)
    
    # Convert to directions
    dir1 = tensor1 / (mag1 + 1e-8)
    dir2 = tensor2 / (mag2 + 1e-8)
    
    # Calculate the angle between the vectors
    dot_product = torch.sum(dir1 * dir2, dim=1, keepdim=True)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)
    
    # Handle small angles
    epsilon = 1e-6
    mask = torch.abs(sin_omega) < epsilon
    
    # Calculate SLERP coefficients
    t1 = torch.where(
        mask,
        1.0 - t,  # Linear for small angles
        torch.sin((1.0 - t) * omega) / sin_omega
    )
    t2 = torch.where(
        mask,
        t,  # Linear for small angles
        torch.sin(t * omega) / sin_omega
    )
    
    # Interpolate directions
    dir_interp = t1 * dir1 + t2 * dir2
    
    # Interpolate magnitudes linearly
    mag_interp = (1 - t) * mag1 + t * mag2
    
    # Combine interpolated direction and magnitude
    interpolated = dir_interp * mag_interp
    
    # Calculate velocity
    # Direction velocity component
    linear_dir_velocity = dir2 - dir1
    spherical_dir_velocity = omega * (
        -torch.cos(omega * (1-t)) * dir1 + 
        torch.cos(omega * t) * dir2
    ) / sin_omega
    dir_velocity = torch.where(mask, linear_dir_velocity, spherical_dir_velocity)
    
    # Magnitude velocity component (constant)
    mag_velocity = mag2 - mag1
    
    # Combine direction and magnitude velocities using product rule:
    # d/dt(r(t)m(t)) = r'(t)m(t) + r(t)m'(t)
    velocity = dir_velocity * mag_interp + dir_interp * mag_velocity
    
    return interpolated, velocity

def autoencoder_process(
    audio, 
    latent_noise_add, 
    latent_noise_interp,
    interp_type,
    n_quantizers
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    if audio.dtype == np.float32:
        audio = torch.from_numpy(audio)
    elif audio.dtype == np.int16:
        audio = torch.from_numpy(audio).float().div(32767)
    elif audio.dtype == np.int32:
        audio = torch.from_numpy(audio).float().div(2147483647)
    else:
        raise ValueError(f"Unsupported audio data type: {audio.dtype}")

    audio = audio.to(device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.transpose(0, 1)

    audio = model.preprocess_audio_for_encoder(audio, in_sr)
    # Note: If you need to do chunked encoding, to reduce VRAM, 
    # then add these arguments to encode_audio and decode_audio: chunked=True, overlap=32, chunk_size=128
    # To turn it off, do chunked=False
    # Optimal overlap and chunk_size values will depend on the model. 
    # See encode_audio & decode_audio in autoencoders.py for more info
    # Get dtype of model
    dtype = next(model.parameters()).dtype

    audio = audio.to(dtype)

    if n_quantizers > 0:
        latents = model.encode_audio(audio, chunked=False, n_quantizers=n_quantizers)
    else:
        latents = model.encode_audio(audio, chunked=False)

    if latent_noise_add > 0:
        latents = latents + torch.randn_like(latents) * latent_noise_add

    # Latent noise interp, like a linear rectified flow schedule
    if latent_noise_interp > 0:
        if interp_type == "linear":
            latents = (1 - latent_noise_interp) * latents + latent_noise_interp * torch.randn_like(latents) * latents.std()
        elif interp_type == "spherical":
            latents, velocity = slerp(latents, torch.randn_like(latents), latent_noise_interp)
        elif interp_type == "v-diffusion":
            alpha, sigma = math.cos(latent_noise_interp * math.pi / 2), math.sin(latent_noise_interp * math.pi / 2)
            latents = alpha * latents + sigma * torch.randn_like(latents)

    audio = model.decode_audio(latents, chunked=False)

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram])

def autoencoder_animate_flow(
    audio, 
    num_steps,
    interp_type
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    if audio.dtype == np.float32:
        audio = torch.from_numpy(audio)
    elif audio.dtype == np.int16:
        audio = torch.from_numpy(audio).float().div(32767)
    elif audio.dtype == np.int32:
        audio = torch.from_numpy(audio).float().div(2147483647)
    else:
        raise ValueError(f"Unsupported audio data type: {audio.dtype}")

    audio = audio.to(device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.transpose(0, 1)

    audio = model.preprocess_audio_for_encoder(audio, in_sr)
    # Note: If you need to do chunked encoding, to reduce VRAM, 
    # then add these arguments to encode_audio and decode_audio: chunked=True, overlap=32, chunk_size=128
    # To turn it off, do chunked=False
    # Optimal overlap and chunk_size values will depend on the model. 
    # See encode_audio & decode_audio in autoencoders.py for more info
    # Get dtype of model
    dtype = next(model.parameters()).dtype

    audio = audio.to(dtype)

    latents = model.encode_audio(audio, chunked=False)
 
    interp_ratios = torch.linspace(0, 1, num_steps, device=device)

    step_spectrograms = []
    step_latents_images = []
    step_velocity_images = []

    noise = torch.randn_like(latents)

    for step in trange(num_steps):
        interp_ratio = interp_ratios[step]
        # Latent noise interp, like a linear rectified flow schedule
        if interp_type == "linear":
            interp_latents = (1 - interp_ratio) * latents + interp_ratio * noise
            velocity = noise - latents
        elif interp_type == "spherical":
            interp_latents, velocity = slerp(latents, noise, interp_ratio)
        elif interp_type == "v-diffusion":
            alpha, sigma = torch.cos(interp_ratio * math.pi / 2), torch.sin(interp_ratio * math.pi / 2)
            interp_latents = alpha * latents + sigma * noise
            velocity = noise * alpha - latents * sigma

        audio = model.decode_audio(interp_latents, chunked=False)

        audio = rearrange(audio, "b d n -> d (b n)")

        audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

        latent_image = tokens_spectrogram_image(interp_latents)
        velocity_image = tokens_spectrogram_image(velocity)

        step_spectrograms.append((audio_spectrogram, f"ratio: {interp_ratio:.2f}"))
        step_latents_images.append((latent_image, f"ratio: {interp_ratio:.2f}"))
        step_velocity_images.append((velocity_image, f"ratio: {interp_ratio:.2f}"))

    return (step_spectrograms, step_latents_images, step_velocity_images)

def create_autoencoder_ui(model_config, in_model):
    global model, sample_size, sample_rate, model_type

    model = in_model
    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]
    model_type = model_config["model_type"]

    is_dac_rvq = "model" in model_config and "bottleneck" in model_config["model"] and model_config["model"]["bottleneck"]["type"] in ["dac_rvq","dac_rvq_vae"]

    if is_dac_rvq:
        n_quantizers = model_config["model"]["bottleneck"]["config"]["n_codebooks"]
    else:
        n_quantizers = 0

    with gr.Blocks() as ui:
        with gr.Tab("Encoding"):
            with gr.Row():
                with gr.Column(scale=6):
                    input_audio = gr.Audio(label="Input audio")
                process_button = gr.Button("Process", variant='primary', scale=1)
            with gr.Row():
                n_quantizers_slider = gr.Slider(minimum=0, maximum=max(n_quantizers, 1), step=1, value=n_quantizers, label="# quantizers", visible=is_dac_rvq)
                add_latent_noise_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.001, value=0.0, label="Add latent noise")
                interp_latent_noise_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, value=0.0, label="Interpolate latent noise")
                interp_type = gr.Dropdown(["v-diffusion", "linear", "spherical"], label="Interpolation type", value="linear")

            output_audio = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)

            process_button.click(fn=autoencoder_process, inputs=[input_audio, add_latent_noise_slider, interp_latent_noise_slider, interp_type, n_quantizers_slider], outputs=[output_audio, audio_spectrogram_output], api_name="process")
        with gr.Tab("Visualize flow"):
            with gr.Row():
                with gr.Column(scale=6):
                    input_audio = gr.Audio(label="Input audio")
                process_button = gr.Button("Process", variant='primary', scale=1)
            with gr.Row():
                n_steps_slider = gr.Slider(minimum=1, maximum=250, step=1, value=10, label="# steps")
                interp_type = gr.Dropdown(["v-diffusion", "linear", "spherical"], label="Interpolation type", value="v-diffusion")

            flow_spec_gallery = gr.Gallery(label="Output spectrograms", show_label=False)
            flow_latent_gallery = gr.Gallery(label="Output latents", show_label=False)
            flow_velocity_gallery = gr.Gallery(label="Output velocities", show_label=False)

            process_button.click(fn=autoencoder_animate_flow, inputs=[input_audio, n_steps_slider, interp_type], outputs=[flow_spec_gallery, flow_latent_gallery, flow_velocity_gallery], api_name="process")

    return ui

import gc
import numpy as np
import gradio as gr
import json
import re
import subprocess
import torch
import torchaudio

from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..interface.aeiou import audio_spectrogram_image
from ..verbose import vprint
from ..inference.generation import generate_diffusion_cond, generate_diffusion_cond_inpaint, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.lora import load_and_apply_loras
from ..models.pretrained import get_pretrained_model
from ..models.utils import copy_state_dict, load_ckpt_state_dict
from ..inference.utils import prepare_audio

from .interfaces.diffusion_cond import create_diffusion_cond_ui
from .interfaces.autoencoder import create_autoencoder_ui

# Mapping of model_type to the attribute name used in the training wrapper
# e.g., DiffusionCondTrainingWrapper stores the model as self.diffusion
WRAPPER_PREFIXES = {
    "diffusion_cond": "diffusion.",
    "diffusion_cond_inpaint": "diffusion.",
    "diffusion_uncond": "diffusion.",
    "diffusion_autoencoder": "diffusion.",
    "autoencoder": "autoencoder.",
    "lm": "lm.",
    "clap": "clap.",
}

def unwrap_state_dict(state_dict, model_type):
    """
    Detect if state_dict is from a wrapped training checkpoint and unwrap it.
    
    Wrapped checkpoints have keys like 'diffusion.model.xxx' or 'diffusion_ema.ema_model.xxx'.
    Unwrapped checkpoints have keys like 'model.xxx' directly.
    
    Returns the unwrapped state_dict.
    """
    prefix = WRAPPER_PREFIXES.get(model_type)
    if prefix is None:
        # Unknown model type, return as-is
        return state_dict
    
    # Check if this is a wrapped checkpoint by looking for the prefix
    has_wrapper_prefix = any(k.startswith(prefix) for k in state_dict.keys())
    
    if not has_wrapper_prefix:
        # Already unwrapped
        return state_dict
    
    vprint(f"Detected wrapped checkpoint, unwrapping with prefix '{prefix}'")
    
    ema_prefix = prefix.replace(".", "_ema.ema_model.")  # e.g., "diffusion." -> "diffusion_ema.ema_model."
    has_ema = any(k.startswith(ema_prefix) for k in state_dict.keys())

    # For diffusion models, the EMA only wraps the inner model (self.diffusion.model),
    # not the conditioner/pretransform. So we need to handle them separately:
    #   EMA keys: diffusion_ema.ema_model.xxx -> model.xxx
    #   Conditioner keys: diffusion.conditioner.xxx -> conditioner.xxx
    #   Pretransform keys: diffusion.pretransform.xxx -> pretransform.xxx
    #
    # For autoencoder models, the EMA wraps the entire autoencoder (self.autoencoder),
    # including encoder, decoder, bottleneck, and pretransform. So we just strip the
    # EMA prefix directly:
    #   EMA keys: autoencoder_ema.ema_model.encoder.xxx -> encoder.xxx

    # Model types where EMA wraps the entire model (not just an inner .model sub-module)
    ema_wraps_whole_model = model_type in ("autoencoder",)

    unwrapped = {}

    if has_ema:
        vprint(f"Using EMA weights with prefix '{ema_prefix}'")
        for k, v in state_dict.items():
            if k.startswith(ema_prefix):
                suffix = k[len(ema_prefix):]
                if ema_wraps_whole_model:
                    # EMA wraps the whole model, just use the suffix directly
                    new_key = suffix
                else:
                    # EMA wraps only .model, so re-add the "model." prefix
                    new_key = "model." + suffix
                unwrapped[new_key] = v

        if not ema_wraps_whole_model:
            # Get conditioner and other non-model weights from the main prefix
            conditioner_prefix = prefix + "conditioner."
            pretransform_prefix = prefix + "pretransform."
            for k, v in state_dict.items():
                if k.startswith(conditioner_prefix):
                    new_key = k[len(prefix):]  # strip "diffusion." to get "conditioner.xxx"
                    unwrapped[new_key] = v
                elif k.startswith(pretransform_prefix):
                    new_key = k[len(prefix):]  # strip to get "pretransform.xxx"
                    unwrapped[new_key] = v
    else:
        # No EMA, just strip the wrapper prefix from all keys
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                unwrapped[new_key] = v
    
    return unwrapped

model = None
model_type = None
sample_rate = 32000
sample_size = 1920000

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False, lora_ckpt_paths=None):
    global model, sample_rate, sample_size, model_type

    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        vprint(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint and unwrap if it's a wrapped training checkpoint
        state_dict = load_ckpt_state_dict(model_ckpt_path)
        state_dict = unwrap_state_dict(state_dict, model_config.get("model_type"))
        copy_state_dict(model, state_dict)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model_type = model_config["model_type"]

    if pretransform_ckpt_path is not None:
        vprint(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        vprint(f"Done loading pretransform")

    if lora_ckpt_paths:
        model.to(device)
        svd_bases_path = model_config.get("svd_bases_path") if model_config else None
        load_and_apply_loras(model, lora_ckpt_paths, model_type, svd_bases_path=svd_bases_path)
    else:
        model.use_lora = False
        model.lora_names = []

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)

    print(f"Done loading model")

    return model, model_config

def generate_uncond(
        steps=250,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        batch_size=1,
        preview_every=None
        ):

    global preview_images

    preview_images = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None
    
    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        # Turn into torch tensor, converting from int16 to float32
        # init_audio = torch.from_numpy(init_audio).float().div(32767)
        if init_audio.dtype == np.float32:
            init_audio = torch.from_numpy(init_audio)
        elif init_audio.dtype == np.int16:
            init_audio = torch.from_numpy(init_audio).float().div(32767)
        elif init_audio.dtype == np.int32:
            init_audio = torch.from_numpy(init_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {init_audio.dtype}")
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0) # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:

            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        # Extract scalar from tensor if needed (samplers pass tensors to avoid GPU sync)
        if isinstance(sigma, torch.Tensor):
            sigma = sigma[0].item() if sigma.dim() > 0 else sigma.item()

        if (current_step - 1) % preview_every == 0:

            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)

            denoised = rearrange(denoised, "b d n -> d (b n)")

            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)

            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    audio = generate_diffusion_uncond(
        model, 
        steps=steps,
        batch_size=batch_size,
        sample_size=input_sample_size,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        callback = progress_callback if preview_every is not None else None
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram, *preview_images])

def generate_lm(
        temperature=1.0,
        top_p=0.95,
        top_k=0,    
        batch_size=1,
        ):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    audio = model.generate_audio(
        batch_size=batch_size,
        max_gen_len = sample_size//model.pretransform.downsampling_ratio,
        conditioning=None,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram])


def create_uncond_sampling_ui(model_config):   
    generate_button = gr.Button("Generate", variant='primary', scale=1)
    
    with gr.Row(equal_height=False):
        with gr.Column():            
            with gr.Row():
                # Steps slider
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")

            with gr.Accordion("Sampler params", open=False):
            
                # Seed
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

            # Sampler params
                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-3m-sde")
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")

            with gr.Accordion("Init audio", open=False):
                init_audio_checkbox = gr.Checkbox(label="Use init audio")
                init_audio_input = gr.Audio(label="Init audio")
                init_noise_level_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, value=0.1, label="Init noise level")

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])
    
    generate_button.click(fn=generate_uncond, 
        inputs=[
            steps_slider, 
            seed_textbox, 
            sampler_type_dropdown, 
            sigma_min_slider, 
            sigma_max_slider,
            init_audio_checkbox,
            init_audio_input,
            init_noise_level_slider,
        ], 
        outputs=[
            audio_output, 
            audio_spectrogram_output
        ], 
        api_name="generate")

def create_diffusion_uncond_ui(model_config):
    with gr.Blocks() as ui:
        create_uncond_sampling_ui(model_config)
    
    return ui

def autoencoder_process(audio, latent_noise, n_quantizers):
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

    if latent_noise > 0:
        latents = latents + torch.randn_like(latents) * latent_noise

    audio = model.decode_audio(latents, chunked=False)

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

# def create_autoencoder_ui(model_config):

#     is_dac_rvq = "model" in model_config and "bottleneck" in model_config["model"] and model_config["model"]["bottleneck"]["type"] in ["dac_rvq","dac_rvq_vae"]

#     if is_dac_rvq:
#         n_quantizers = model_config["model"]["bottleneck"]["config"]["n_codebooks"]
#     else:
#         n_quantizers = 0

#     with gr.Blocks() as ui:
#         input_audio = gr.Audio(label="Input audio")
#         output_audio = gr.Audio(label="Output audio", interactive=False)
#         n_quantizers_slider = gr.Slider(minimum=1, maximum=n_quantizers, step=1, value=n_quantizers, label="# quantizers", visible=is_dac_rvq)
#         latent_noise_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.001, value=0.0, label="Add latent noise")
#         process_button = gr.Button("Process", variant='primary', scale=1)
#         process_button.click(fn=autoencoder_process, inputs=[input_audio, latent_noise_slider, n_quantizers_slider], outputs=output_audio, api_name="process")

#     return ui

def create_lm_ui(model_config):
    with gr.Blocks() as ui:
        output_audio = gr.Audio(label="Output audio", interactive=False)
        audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)

        # Sampling params
        with gr.Row():
            temperature_slider = gr.Slider(minimum=0, maximum=5, step=0.01, value=1.0, label="Temperature")
            top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.95, label="Top p")
            top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Top k")

        generate_button = gr.Button("Generate", variant='primary', scale=1)
        generate_button.click(
            fn=generate_lm, 
            inputs=[
                temperature_slider, 
                top_p_slider, 
                top_k_slider
            ], 
            outputs=[output_audio, audio_spectrogram_output],
            api_name="generate"
        )

    return ui

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False, gradio_title="", lora_ckpt_paths=None, default_prompt=None):
    assert (pretrained_name is not None) ^ (model_config_path is not None and ckpt_path is not None), "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    if model_config_path is not None:
        # Load config from json file
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name, pretransform_ckpt_path=pretransform_ckpt_path, model_half=model_half, lora_ckpt_paths=lora_ckpt_paths, device=device)
    
    if model_type == "diffusion_cond" or model_type == "diffusion_cond_inpaint":
        ui = create_diffusion_cond_ui(model_config, model, in_model_half=model_half, gradio_title=gradio_title, default_prompt=default_prompt)
    elif model_type == "diffusion_uncond":
        ui = create_diffusion_uncond_ui(model_config)
    elif model_type == "autoencoder" or model_type == "diffusion_autoencoder":
        ui = create_autoencoder_ui(model_config, in_model=model)
    elif model_type == "lm":
        ui = create_lm_ui(model_config)
        
    return ui
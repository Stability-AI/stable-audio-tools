import numpy as np
import gradio as gr
import json 
import torch
import torchaudio

from torch.nn import functional as F

from einops import rearrange

from ..inference.generation import generate_diffusion_cond
from ..models.factory import create_model_from_config
from ..inference.utils import prepare_audio

model = None
sample_rate = 32000
sample_size = 1920000

def load_model(model_config, model_ckpt_path, device="cuda"):
    global model, sample_rate, sample_size
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    print(f"Creating model from config")
    model = create_model_from_config(model_config)

    model.to(device)

    print(f"Loading model checkpoint from {model_ckpt_path}")
    # Load checkpoint
    model.load_state_dict(torch.load(model_ckpt_path)["state_dict"], strict=False)
    print(f"Done loading model")

def generate(
        prompt,
        seconds_start=0,
        seconds_total=30,
        cfg_scale=6.0,
        steps=250,
        seed=-1,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.5,
        sigma_max=50,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        batch_size=1,
        ):
    # Return fake stereo audio

    conditioning = [{"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size

    #Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None
    
    if init_audio is not None:
        in_sr, init_audio = init_audio

        # Turn into torch tensor, converting from int16 to float32
        init_audio = torch.from_numpy(init_audio).float().div(32767).transpose(0, 1)

        init_audio = (in_sr, init_audio)

    audio = generate_diffusion_cond(
        model, 
        conditioning=conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        sample_size=sample_size,
        sample_rate=sample_rate,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav" 

def create_sampling_ui():
    with gr.Row():
        prompt = gr.Textbox(show_label=False, placeholder="Prompt", scale=6)
        generate_button = gr.Button("Generate", variant='primary', scale=1)
    
    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row():
                # Timing controls
                seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds start")
                seconds_total_slider = gr.Slider(minimum=0, maximum=512, step=1, value=60, label="Seconds total")
            
            with gr.Row():
                # Steps slider
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=200, label="Steps")

                # CFG scale 
                cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=7.0, label="CFG scale")

            with gr.Accordion("Sampler params", open=False):
            
                # Seed
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

            # Sampler params
                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-2m-sde")
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.95, label="Sigma min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=200.0, step=0.1, value=77, label="Sigma max")

            with gr.Accordion("Init audio", open=False):
                init_audio_checkbox = gr.Checkbox(label="Use init audio")
                init_audio_input = gr.Audio(label="Init audio")
                init_noise_level_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, value=0.1, label="Init noise level")

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", scale=6, interactive=False)
            # audio_spectrogram_output = gr.Image(label="Output spectrogram", scale=6, interactive=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])
    
    generate_button.click(fn=generate, inputs=[
        prompt, 
        seconds_start_slider, 
        seconds_total_slider, 
        cfg_scale_slider, 
        steps_slider, 
        seed_textbox, 
        sampler_type_dropdown, 
        sigma_min_slider, 
        sigma_max_slider,
        init_audio_checkbox,
        init_audio_input,
        init_noise_level_slider,
        ], outputs=audio_output, api_name="generate")


def create_txt2audio_ui():
    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_sampling_ui()
    
    return ui

def autoencoder_process(audio):
    # Return fake stereo audio

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.transpose(0, 1)

    audio_length = audio.shape[1]

    # Pad to multiple of model's downsampling ratio
    pad_length = (model.downsampling_ratio - (audio_length % model.downsampling_ratio)) % model.downsampling_ratio
    audio = F.pad(audio, (0, pad_length))

    audio = prepare_audio(audio, in_sr=in_sr, target_sr=sample_rate, target_length=audio.shape[1], target_channels=model.io_channels, device=device)

    latents = model.encode(audio)
    audio = model.decode(latents)

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def create_autoencoder_ui():
    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=autoencoder_process, inputs=[input_audio], outputs=output_audio, api_name="process")

    return ui


def create_ui(model_config, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model(model_config, ckpt_path, device=device)
    
    model_type = model_config["model_type"]

    if model_type == "diffusion_cond":
        ui = create_txt2audio_ui()
    elif model_type == "diffusion_uncond":
        raise NotImplementedError("Unconditional diffusion is not supported yet")
    elif model_type == "autoencoder":
        ui = create_autoencoder_ui()
        

    return ui
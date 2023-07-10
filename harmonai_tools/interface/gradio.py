import numpy as np
import gradio as gr
import json 
import torch
import torchaudio

from einops import rearrange

from ..inference.generation import generate_diffusion_cond
from ..models.factory import create_model_from_config

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
        device="cuda",
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
    prompt = gr.Textbox(label="Prompt")
    
    with gr.Row():
        # Timing controls
        seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds start")
        seconds_total_slider = gr.Slider(minimum=0, maximum=512, step=1, value=60, label="Seconds total")
        
        # Steps slider
        steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=250, label="Steps")

        # CFG scale 
        cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=6.0, label="CFG scale")

    with gr.Accordion("Sampler params", open=False):
    
        # Seed
        seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

    # Sampler params
        with gr.Row():
            sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-2m-sde")
            sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.5, label="Sigma min")
            sigma_max_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=50.0, label="Sigma max")

    with gr.Accordion("Init audio", open=False):
        init_audio_checkbox = gr.Checkbox(label="Use init audio")
        init_audio_input = gr.Audio(label="Init audio")
        init_noise_level_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, value=0.1, label="Init noise level")

    audio_output = gr.Audio(label="Output audio")
    
    generate_button = gr.Button("Generate")
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


def create_ui(model_config, ckpt_path):
    load_model(model_config, ckpt_path)
    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_sampling_ui()
        
    return ui
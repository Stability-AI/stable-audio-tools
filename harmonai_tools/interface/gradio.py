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
    model.load_state_dict(torch.load(model_ckpt_path)["state_dict"])
    print(f"Done loading model")

def generate(
        prompt,
        seconds_start=0,
        seconds_total=30,
        cfg_scale=6.0,
        steps=250,
        batch_size=1,
        ):
    # Return fake stereo audio

    conditioning = [{"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size

    audio = generate_diffusion_cond(
        model, 
        conditioning=conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        sample_size=sample_size,
        device="cuda"
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav" 

def create_ui(model_config, ckpt_path):
    load_model(model_config, ckpt_path)
    with gr.Blocks() as ui:
        prompt = gr.Textbox(label="Prompt")
        
        # Timing controls
        seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds start")
        seconds_total_slider = gr.Slider(minimum=0, maximum=512, step=1, value=60, label="Seconds total")
        
        # Steps slider
        steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=250, label="Steps")

        # CFG scale 
        cfg_scale_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=6.0, label="CFG scale")

        audio_output = gr.Audio(label="Output audio")
        
        generate_button = gr.Button("Generate")
        generate_button.click(fn=generate, inputs=[prompt, seconds_start_slider, seconds_total_slider, cfg_scale_slider, steps_slider], outputs=audio_output, api_name="generate")

    return ui
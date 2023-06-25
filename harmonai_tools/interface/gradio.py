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

def load_model(model_config_path, model_ckpt_path, device="cuda"):
    global model, sample_rate, sample_size
    print(f"Loading model config from {model_config_path}")
    # Load config from json file
    with open(model_config_path) as f:
        model_config = json.load(f)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    print(f"Creating model from config")
    model = create_model_from_config(model_config)

    model.to(device)

    print(f"Loading model checkpoint from {model_ckpt_path}")
    # Load checkpoint
    model.load_state_dict(torch.load(model_ckpt_path)["state_dict"])
    print(f"Done loading model")

def generate(prompt):
    # Return fake stereo audio

    conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": 30}]

    audio = generate_diffusion_cond(
        model, 
        conditioning=conditioning,
        batch_size=1,
        sample_size=sample_size,
        device="cuda"
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav" 

def create_ui():
    with gr.Blocks() as ui:
        with gr.Row(label="Model config"):
            model_config = gr.Textbox(label="Model config")
            model_ckpt = gr.Textbox(label="Model checkpoint")
            load_model_button = gr.Button("Load model")
            load_model_button.click(fn=load_model, inputs=[model_config, model_ckpt])
        prompt = gr.Textbox(label="Prompt")
        audio_output = gr.Audio(label="Output audio")
        generate_button = gr.Button("Generate")
        generate_button.click(fn=generate, inputs=prompt, outputs=audio_output, api_name="generate")

    return ui
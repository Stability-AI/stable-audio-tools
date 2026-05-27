import gc
import random
import numpy as np
import gradio as gr
import json
import re
import subprocess
import torch
import torchaudio
import threading 
import os, time, math

from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..aeiou import audio_spectrogram_image
from ...verbose import vprint
from ...inference.generation import generate_diffusion_cond, generate_diffusion_cond_inpaint #, generate_diffusion_uncond
from ...inference.sampling import LogSNRShift, FluxDistributionShift, DistributionShift, IdentityDistributionShift
#from ..models.factory import create_model_from_config
#from ..models.pretrained import get_pretrained_model
#from ..models.utils import copy_state_dict, load_ckpt_state_dict
from ...inference.utils import prepare_audio
from ...models.lora import set_lora_strength, has_lora, get_lora_count

model = None
model_type = None
sample_size = 2097152
sample_rate = 44100
model_half = True
diffusion_objective = None
mask_padding_attention = False
n_loras = 0

# when using a prompt in a filename
def condense_prompt(prompt):
    pattern = r'[\\/:*?"<>|]'
    # Replace special characters with hyphens
    prompt = re.sub(pattern, '-', prompt)
    # set a character limit 
    prompt = prompt[:150]
    # zero length prompts may lead to filenames (ie ".wav") which seem cause problems with gradio
    if len(prompt)==0:
        prompt = "_"
    return prompt

def generate_cond(
        prompt,
        negative_prompt=None,
        seconds_start=0,
        seconds_total=30,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        rho=1.0,
        cfg_interval_min=0.0,
        cfg_interval_max=1.0,
        cfg_rescale=0.0,
        cfg_norm_threshold=0.0,
        apg_scale=1.0,
        file_format="wav",
        file_naming="verbose",
        cut_to_seconds_total=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_maskstart=None,
        mask_maskend=None,
        inpaint_audio=None,
        init_audio_type="Init audio",
        inversion_steps=100,
        inversion_gamma=0.3,
        inversion_unconditional=False,
        adapt_duration_to_conditioning=False,
        duration_padding_sec=6.0,
        use_effective_length_for_schedule=False,
        batch_size=1,
        dist_shift=None,
        *lora_args
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Return fake stereo audio
    conditioning_dict = {"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}

    conditioning = [conditioning_dict] * batch_size

    if negative_prompt:
        negative_conditioning_dict = {"prompt": negative_prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}

        negative_conditioning = [negative_conditioning_dict] * batch_size
    else:
        negative_conditioning = None
        
    #Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)
    # if seed is -1, define the seed value now, randomly, so we can save it in the filename
    if(seed==-1):
        seed = np.random.randint(0, 99999)

    # Parse per-LoRA controls from trailing args
    # Each LoRA has 4 controls: strength, interval_min, interval_max, layer_filter
    lora_configs = None
    if n_loras > 0 and len(lora_args) >= n_loras * 4:
        lora_configs = []
        for i in range(n_loras):
            off = i * 4
            strength = lora_args[off]
            interval_min = lora_args[off + 1]
            interval_max = lora_args[off + 2]
            layer_filter = lora_args[off + 3]
            set_lora_strength(model.model, strength, lora_index=i)
            set_lora_strength(model.conditioner, strength, lora_index=i)
            lora_configs.append({
                "lora_index": i,
                "interval": (interval_min, interval_max),
                "layer_filter": layer_filter,
            })

    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio

        if init_audio.dtype == np.float32:
            init_audio = torch.from_numpy(init_audio)
        elif init_audio.dtype == np.int16:
            init_audio = torch.from_numpy(init_audio).float().div(32767)
        elif init_audio.dtype == np.int32:
            init_audio = torch.from_numpy(init_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {init_audio.dtype}")

        if model_half:
            init_audio = init_audio.to(torch.float16)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0) # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device).to(init_audio.dtype)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:

            #input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length
            init_audio = init_audio[:, :sample_size]

        init_audio = (sample_rate, init_audio)

    if inpaint_audio is not None:
        in_sr, inpaint_audio = inpaint_audio
        
        if inpaint_audio.dtype == np.float32:
            inpaint_audio = torch.from_numpy(inpaint_audio)
        elif inpaint_audio.dtype == np.int16:
            inpaint_audio = torch.from_numpy(inpaint_audio).float().div(32767)
        elif inpaint_audio.dtype == np.int32:
            inpaint_audio = torch.from_numpy(inpaint_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {inpaint_audio.dtype}")

        if model_half:
            inpaint_audio = inpaint_audio.to(torch.float16)
        
        if inpaint_audio.dim() == 1:
            inpaint_audio = inpaint_audio.unsqueeze(0) # [1, n]
        elif inpaint_audio.dim() == 2:
            inpaint_audio = inpaint_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(inpaint_audio.device).to(inpaint_audio.dtype)
            inpaint_audio = resample_tf(inpaint_audio)

        audio_length = inpaint_audio.shape[-1]

        if audio_length > sample_size:

            #input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length
            inpaint_audio = inpaint_audio[:, :sample_size]

        inpaint_audio = (sample_rate, inpaint_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        t = callback_info["t"]
        sigma = callback_info["sigma"]

        # Extract scalar from tensor if needed (samplers pass tensors to avoid GPU sync)
        if isinstance(t, torch.Tensor):
            t = t[0].item() if t.dim() > 0 else t.item()
        if isinstance(sigma, torch.Tensor):
            sigma = sigma[0].item() if sigma.dim() > 0 else sigma.item()

        if diffusion_objective == "v":
            alphas, sigmas = math.cos(t * math.pi / 2), math.sin(t * math.pi / 2)
            log_snr = math.log((alphas / sigmas) + 1e-6)
        elif diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            log_snr = math.log(((1 - sigma) / sigma) + 1e-6)

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f} logSNR={log_snr:.3f}"))

    if init_audio_type == "RF-Inversion":
        inversion_params = {
            "inversion_steps": inversion_steps,
            "inversion_gamma": inversion_gamma,
            "inversion_unconditional": inversion_unconditional,
            "inversion_cfg_scale": 1.0,
            "inversion_sigma_max": 1.0
        }
    else:
        inversion_params = None

    generate_args = {
        "model": model,
        "conditioning": conditioning,
        "negative_conditioning": negative_conditioning,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "cfg_interval": (cfg_interval_min, cfg_interval_max),
        "lora_configs": lora_configs,
        "batch_size": batch_size,
        "sample_size": input_sample_size,
        "seed": seed,
        "device": device,
        "sampler_type": sampler_type,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "init_audio": init_audio,
        "init_noise_level": init_noise_level,
        "callback": progress_callback if preview_every is not None else None,
        "scale_phi": cfg_rescale,
        "cfg_norm_threshold": cfg_norm_threshold,
        "apg_scale": apg_scale,
        "rho": rho,
        "adapt_duration_to_conditioning": adapt_duration_to_conditioning,
        "duration_padding_sec": duration_padding_sec,
        "use_effective_length_for_schedule": use_effective_length_for_schedule,
        "mask_padding_attention": mask_padding_attention,
        "dist_shift": dist_shift,
    }

     # If inpainting, send mask args
    # This will definitely change in the future
    if model_type == "diffusion_cond":

        generate_args["inversion_params"] = inversion_params

        # Do the audio generation
        audio = generate_diffusion_cond(**generate_args)

    elif model_type == "diffusion_cond_inpaint":

        if inpaint_audio is not None:
            generate_args.update({
                "inpaint_audio": inpaint_audio,
                "inpaint_mask_start_seconds": mask_maskstart,
                "inpaint_mask_end_seconds": mask_maskend,
            })

        audio = generate_diffusion_cond_inpaint(**generate_args)

    # Filenaming convention
    prompt_condensed = condense_prompt(prompt) 
    if file_naming=="verbose":
        basename = prompt_condensed
        if negative_prompt:
            basename += ".neg-%s" % condense_prompt(negative_prompt)
        basename += ".cfg%s" % (cfg_scale)
        if sigma_max not in [1.0, 100.0]: 
            # this is a common parameter to tweak, if it's not a default value, put it in the verbose filename
            basename += ".smx%s" % sigma_max
        basename += ".%s" % seed
    elif file_naming=="prompt":
        basename = prompt_condensed
    else:
        # simple e.g. "output.wav"
        basename = "output" 

    if file_format:
        filename_extension = file_format.split(" ")[0].lower()
    else: 
        filename_extension = "wav"
    output_filename = "%s.%s" % (basename, filename_extension)
    output_wav = "%s.wav" % basename

    # Cut the extra silence off the end, if the user requested a smaller seconds_total
    if cut_to_seconds_total:
        audio = audio[:,:,:seconds_total*sample_rate]

    # Encode the audio to WAV format
    audio = rearrange(audio, "b d n -> d (b n)")
    #audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # save as wav file
    torchaudio.save(output_wav, audio, sample_rate)

    # If file_format is other than wav, convert to other file format
    cmd = ""
    if file_format == "m4a aac_he_v2 32k":
        # note: need to compile ffmpeg with --enable-libfdk_aac
        cmd = f"ffmpeg -i \"{output_wav}\" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 32k -y \"{output_filename}\""
    elif file_format == "m4a aac_he_v2 64k":
        cmd = f"ffmpeg -i \"{output_wav}\" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 64k -y \"{output_filename}\""
    elif file_format == "flac":
        cmd = f"ffmpeg -i \"{output_wav}\" -y \"{output_filename}\""
    elif file_format == "mp3 320k":
        cmd = f"ffmpeg -i \"{output_wav}\" -b:a 320k -y \"{output_filename}\""
    elif file_format == "mp3 128k":
        cmd = f"ffmpeg -i \"{output_wav}\" -b:a 128k -y \"{output_filename}\""
    elif file_format == "mp3 v0":
        cmd = f"ffmpeg -i \"{output_wav}\" -q:a 0 -y \"{output_filename}\""
    else: # wav
        pass
    if cmd:
        cmd += " -loglevel error" # make output less verbose in the cmd window
        subprocess.run(cmd, shell=True, check=True)
    
    # Let's look at a nice spectrogram too
    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    # Asynchronously delete the files after returning the output file, so as to prevent clutter in the directory
    if file_naming in ["verbose", "prompt"]:
        delete_files_async([output_wav, output_filename], 30)

    return (output_filename, [audio_spectrogram, *preview_images])

#  Asynchronously delete the given list of filenames after delay seconds. Sets up thread that sleeps for delay then deletes. 
def delete_files_async(filenames, delay):
    def delete_files_after_delay(filenames, delay):
        time.sleep(delay)  # Wait for the specified delay
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)  # Delete the file
    threading.Thread(target=delete_files_after_delay, args=(filenames, delay)).start() 

def create_sampling_ui(model_config, default_prompt=None):
    global diffusion_objective, n_loras, mask_padding_attention
    has_inpainting = model_config["model_type"] == "diffusion_cond_inpaint"

    model_conditioning_config = model_config["model"].get("conditioning", None)

    diffusion_objective = model.diffusion_objective
    is_rf = diffusion_objective == "rectified_flow"
    is_rf_denoiser = diffusion_objective == "rf_denoiser" # includes ARC models
    is_v = diffusion_objective == "v"

    # Read from model wrapper (with fallback to training config for backward compat)
    trained_with_effective_length = getattr(model, 'use_effective_length_for_schedule', False)
    trained_with_masking = getattr(model, 'mask_padding_attention', False)
    if not trained_with_effective_length or not trained_with_masking:
        training_config = model_config.get("training", {})
        if not trained_with_effective_length:
            trained_with_effective_length = training_config.get("use_effective_length_for_schedule", False)
        if not trained_with_masking:
            trained_with_masking = training_config.get("mask_padding_attention", False)
    mask_padding_attention = trained_with_masking

    # Extract default dist_shift params from model's sampling_dist_shift
    default_sampling_dist_shift = getattr(model, 'sampling_dist_shift', None)
    default_dist_shift_type = "LogSNR"
    default_logsnr_params = {"anchor_length": 2000, "anchor_logsnr": -6.2, "rate": 0.0, "logsnr_end": 2.0}
    default_flux_params = {"min_length": 256, "max_length": 4096, "alpha_min": 6.93, "alpha_max": 6.93}
    default_full_params = {"base_shift": 0.5, "max_shift": 1.15, "min_length": 256, "max_length": 4096}

    if isinstance(default_sampling_dist_shift, LogSNRShift):
        default_dist_shift_type = "LogSNR"
        default_logsnr_params = {
            "anchor_length": getattr(default_sampling_dist_shift, 'anchor_length', 2000),
            "anchor_logsnr": getattr(default_sampling_dist_shift, 'anchor_logsnr', -6.2),
            "rate": getattr(default_sampling_dist_shift, 'rate', 0.0),
            "logsnr_end": getattr(default_sampling_dist_shift, 'logsnr_end', 2.0),
        }
    elif isinstance(default_sampling_dist_shift, FluxDistributionShift):
        default_dist_shift_type = "Flux"
        default_flux_params = {
            "min_length": default_sampling_dist_shift.min_length,
            "max_length": default_sampling_dist_shift.max_length,
            "alpha_min": default_sampling_dist_shift.alpha_min,
            "alpha_max": default_sampling_dist_shift.alpha_max,
        }
    elif isinstance(default_sampling_dist_shift, DistributionShift):
        default_dist_shift_type = "Full"
        default_full_params = {
            "base_shift": default_sampling_dist_shift.base_shift,
            "max_shift": default_sampling_dist_shift.max_shift,
            "min_length": default_sampling_dist_shift.min_length,
            "max_length": default_sampling_dist_shift.max_length,
        }
    elif default_sampling_dist_shift is None:
        default_dist_shift_type = "None"

    has_seconds_start = False
    has_seconds_total = False

    use_lora = has_lora(model)

    lora_names = getattr(model, 'lora_names', [])
    n_loras = len(lora_names)

    # Use provided default prompt, or empty string if not set
    if default_prompt is None:
        default_prompt = ""

    if model_conditioning_config is not None:
        for conditioning_config in model_conditioning_config["configs"]:
            if conditioning_config["id"] == "seconds_start":
                has_seconds_start = True
            if conditioning_config["id"] == "seconds_total":
                has_seconds_total = True

    with gr.Row():
        with gr.Column(scale=6):
            prompt = gr.Textbox(show_label=False, placeholder="Prompt", value=default_prompt)
            negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt")
        generate_button = gr.Button("Generate", variant='primary', scale=1)

    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row(visible = has_seconds_start or has_seconds_total):
                # Timing controls
                seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds start", visible=has_seconds_start)
                seconds_total_slider = gr.Slider(minimum=0, maximum=512, step=1, value=sample_size//sample_rate, label="Seconds total", visible=has_seconds_total)
            
            with gr.Row():
                # Steps slider
                if is_rf:
                    default_steps = 50
                elif is_rf_denoiser:
                    default_steps = 8
                else:
                    default_steps = 100
                    
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=default_steps, label="Steps")
                # CFG scale 
                default_cfg_scale = 1.0 if is_rf_denoiser else 7.0
                cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=default_cfg_scale, label="CFG scale")

            # Per-LoRA controls (dynamic based on number of loaded LoRAs)
            lora_ui_inputs = []
            if use_lora and lora_names:
                for i, lora_name in enumerate(lora_names):
                    with gr.Accordion("LoRA {}: {}".format(i + 1, lora_name), open=(i == 0)):
                        with gr.Row():
                            strength = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=1.0, label="Strength")
                            int_min = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Interval min")
                            int_max = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="Interval max")
                            lyr_filt = gr.Textbox(label="Layer filter", placeholder="")
                        lora_ui_inputs.extend([strength, int_min, int_max, lyr_filt])


            with gr.Accordion("Sampler params", open=False):
                with gr.Row():
                    # Seed
                    seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

                    cfg_interval_min_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG interval min")
                    cfg_interval_max_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=1.0, label="CFG interval max")

                with gr.Row():
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG rescale amount")
                    cfg_norm_threshold = gr.Slider(minimum=0.0, maximum=100, step=0.1, value=0.0, label="CFG norm threshold")
                    apg_scale_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=1.0, label="APG scale", info="1.0=full APG, 0.0=vanilla CFG")

                with gr.Row():
                    # Sampler params
                    if is_rf:
                        sampler_types = ["euler", "rk4", "dpmpp"]
                        default_sampler_type = "dpmpp"
                        sigma_max_max = 1.0
                        sigma_max_default = 1.0
                    elif is_rf_denoiser:
                        sampler_types = ["pingpong"]
                        default_sampler_type = "pingpong"
                        sigma_max_max = 1.0
                        sigma_max_default = 1.0
                    else:
                        sampler_types = ["dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-adaptive", "k-dpm-fast", "v-ddim", "v-ddim-cfgpp"]
                        default_sampler_type = "dpmpp-3m-sde"
                        sigma_max_max = 1000.0
                        sigma_max_default = 100.0
                        
                    sampler_type_dropdown = gr.Dropdown(sampler_types, label="Sampler type", value=default_sampler_type)
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.01, label="Sigma min", visible=is_v)
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=sigma_max_max, step=0.1, value=sigma_max_default, label="Sigma max", visible=True)
                    rho_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.01, value=1.0, label="Sigma curve strength", visible=is_v)

                with gr.Row():
                    adapt_duration_checkbox = gr.Checkbox(label="Adapt duration to conditioning", value=trained_with_masking, info="Generate at shorter sequence length based on seconds_total + padding")
                    duration_padding_slider = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, value=6.0, label="Duration padding (sec)")
                    use_effective_length_checkbox = gr.Checkbox(label="Use effective length for distribution shift", value=trained_with_effective_length, info="Adjust timestep schedule based on seconds_total")

                def build_dist_shift(shift_type, p1, p2, p3, p4):
                    """Build dist_shift from type + 4 params (meaning depends on type)."""
                    if shift_type == "LogSNR":
                        return LogSNRShift(anchor_length=int(p1), anchor_logsnr=p2, rate=p3, logsnr_end=p4)
                    elif shift_type == "Flux":
                        return FluxDistributionShift(min_length=int(p1), max_length=int(p2), alpha_min=p3, alpha_max=p4)
                    elif shift_type == "Full":
                        return DistributionShift(base_shift=p1, max_shift=p2, min_length=int(p3), max_length=int(p4))
                    return IdentityDistributionShift()  # "None" = no shift

                dist_shift_state = gr.State(value=default_sampling_dist_shift)

                with gr.Row(visible=is_rf or is_rf_denoiser):
                    dist_shift_type_dropdown = gr.Dropdown(
                        ["LogSNR", "Flux", "Full", "None"],
                        label="Sampling schedule shift",
                        value=default_dist_shift_type,
                        info="Distribution shift applied to sampling timesteps"
                    )
                with gr.Row(visible=(is_rf or is_rf_denoiser) and default_dist_shift_type == "LogSNR") as logsnr_params_row:
                    logsnr_anchor_length_slider = gr.Slider(minimum=100, maximum=10000, step=100, value=default_logsnr_params["anchor_length"], label="Anchor length")
                    logsnr_anchor_logsnr_slider = gr.Slider(minimum=-12.0, maximum=0.0, step=0.1, value=default_logsnr_params["anchor_logsnr"], label="Anchor log-SNR")
                    logsnr_rate_slider = gr.Slider(minimum=-2.0, maximum=2.0, step=0.1, value=default_logsnr_params["rate"], label="Rate")
                    logsnr_end_slider = gr.Slider(minimum=-2.0, maximum=6.0, step=0.1, value=default_logsnr_params["logsnr_end"], label="log-SNR end")
                with gr.Row(visible=(is_rf or is_rf_denoiser) and default_dist_shift_type == "Flux") as flux_params_row:
                    flux_min_length_slider = gr.Slider(minimum=1, maximum=10000, step=1, value=default_flux_params["min_length"], label="Min seq len")
                    flux_max_length_slider = gr.Slider(minimum=1, maximum=10000, step=1, value=default_flux_params["max_length"], label="Max seq len")
                    flux_alpha_min_slider = gr.Slider(minimum=0.1, maximum=20.0, step=0.1, value=default_flux_params["alpha_min"], label="Alpha min")
                    flux_alpha_max_slider = gr.Slider(minimum=0.1, maximum=20.0, step=0.1, value=default_flux_params["alpha_max"], label="Alpha max")
                with gr.Row(visible=(is_rf or is_rf_denoiser) and default_dist_shift_type == "Full") as full_params_row:
                    full_base_shift_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.05, value=default_full_params["base_shift"], label="Base shift")
                    full_max_shift_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.05, value=default_full_params["max_shift"], label="Max shift")
                    full_min_length_slider = gr.Slider(minimum=1, maximum=10000, step=1, value=default_full_params["min_length"], label="Min length")
                    full_max_length_slider = gr.Slider(minimum=1, maximum=10000, step=1, value=default_full_params["max_length"], label="Max length")

                # Per-type slider groups for wiring to state
                logsnr_sliders = [logsnr_anchor_length_slider, logsnr_anchor_logsnr_slider, logsnr_rate_slider, logsnr_end_slider]
                flux_sliders = [flux_min_length_slider, flux_max_length_slider, flux_alpha_min_slider, flux_alpha_max_slider]
                full_sliders = [full_base_shift_slider, full_max_shift_slider, full_min_length_slider, full_max_length_slider]
                all_dist_shift_inputs = [dist_shift_type_dropdown] + logsnr_sliders + flux_sliders + full_sliders

                def update_dist_shift_state(shift_type, *params):
                    """Route the 4 relevant params to build_dist_shift based on type."""
                    type_to_slice = {"LogSNR": params[0:4], "Flux": params[4:8], "Full": params[8:12]}
                    p = type_to_slice.get(shift_type, (0, 0, 0, 0))
                    return (
                        build_dist_shift(shift_type, *p),
                        gr.update(visible=((is_rf or is_rf_denoiser) and (shift_type == "LogSNR"))),
                        gr.update(visible=((is_rf or is_rf_denoiser) and (shift_type == "Flux"))),
                        gr.update(visible=((is_rf or is_rf_denoiser) and (shift_type == "Full"))),
                    )

                for component in all_dist_shift_inputs:
                    component.change(
                        update_dist_shift_state,
                        inputs=all_dist_shift_inputs,
                        outputs=[dist_shift_state, logsnr_params_row, flux_params_row, full_params_row],
                    )

            # Hidden state for batch_size (no UI control, but needed for function signature)
            batch_size_state = gr.State(value=1)

            with gr.Accordion("Output params", open=False):
                # Output params
                with gr.Row():
                    file_format_dropdown = gr.Dropdown(["wav", "flac", "mp3 320k", "mp3 v0", "mp3 128k", "m4a aac_he_v2 64k", "m4a aac_he_v2 32k"], label="File format", value="wav")
                    file_naming_dropdown = gr.Dropdown(["verbose", "prompt", "output.wav"], label="File naming", value="verbose") # ,"prompt","verbose"
                    preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Spec Preview Every")
                
                    cut_to_seconds_total_checkbox = gr.Checkbox(label="Cut to seconds total", value=True)
                    autoplay_checkbox = gr.Checkbox(label="Autoplay", value=False, elem_id="autoplay")
                    infinite_radio_checkbox = gr.Checkbox(label="Infinite Radio", value=False, elem_id="infinite-radio")
                    automatic_download_checkbox = gr.Checkbox(label="Auto Download", value=False, elem_id="automatic-download")

            # Default generation tab
            with gr.Accordion("Init audio", open=False):
                init_audio_input = gr.Audio(label="Init audio", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
                min_noise_level = 0.1 if is_v else 0.01
                max_noise_level = 100.0 if is_v else 1.0
                default_noise_level = 8 if is_v else 0.9 # roughly halfway style transfer values
                if is_rf:
                    choices = ["Init audio","RF-Inversion"]
                else:
                    choices = ["Init audio"]

                init_audio_type_radio = gr.Radio(label="Techniques", choices=choices, value=choices[0], visible=len(choices)>1)
                with gr.Column(visible=True) as interface_a:
                    init_noise_level_slider = gr.Slider(minimum=min_noise_level, maximum=max_noise_level, step=0.01, value=default_noise_level, label="Init noise level")
                with gr.Column(visible=False) as interface_b:
                    inversion_steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Inversion Steps")
                    inversion_gamma_slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0, label="Gamma", visible=True)
                    inversion_unconditional_checkbox = gr.Checkbox(label="Unconditional", value=False)
                    gr.HTML("<div style='opacity: 0.5; padding: 0px'>For reproduction, try empty prompt, cfg 1, gamma .3<br>\
                        For prompt re-stylization, try cfg 1-7, gamma 0-.15, unconditional</div>")
                def init_audio_type_switch(choice):
                    return (
                        gr.update(visible=(choice == "Init audio")),
                        gr.update(visible=(choice == "RF-Inversion"))
                    )
                init_audio_type_radio.change(init_audio_type_switch, inputs=init_audio_type_radio, outputs=[interface_a, interface_b])

            with gr.Accordion("Inpainting", open=False, visible=has_inpainting):
                inpaint_audio_input = gr.Audio(label="Inpaint audio", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
                mask_maskstart_slider = gr.Slider(minimum=0.0, maximum=sample_size//sample_rate, step=0.1, value=0, label="Mask Start (sec)")
                mask_maskend_slider = gr.Slider(minimum=0.0, maximum=sample_size//sample_rate, step=0.1, value=sample_size//sample_rate, label="Mask End (sec)")

                # Update inpainting slider ranges when seconds_total changes.
                # Only seconds_total is an input — reading the mask sliders here would cause
                # validation errors since their values may exceed the about-to-be-reduced maximum.
                def update_inpaint_sliders(seconds_total):
                    max_val = max(seconds_total, 1)
                    return (
                        gr.update(maximum=max_val),
                        gr.update(maximum=max_val, value=max_val),
                    )
                seconds_total_slider.change(update_inpaint_sliders, inputs=[seconds_total_slider], outputs=[mask_maskstart_slider, mask_maskend_slider])

            inputs = [
                prompt,
                negative_prompt,
                seconds_start_slider,
                seconds_total_slider,
                cfg_scale_slider,
                steps_slider,
                preview_every_slider,
                seed_textbox,
                sampler_type_dropdown,
                sigma_min_slider,
                sigma_max_slider,
                rho_slider,
                cfg_interval_min_slider,
                cfg_interval_max_slider,
                cfg_rescale_slider,
                cfg_norm_threshold,
                apg_scale_slider,
                file_format_dropdown,
                file_naming_dropdown,
                cut_to_seconds_total_checkbox,
                init_audio_input,
                init_noise_level_slider,
                mask_maskstart_slider,
                mask_maskend_slider,
                inpaint_audio_input,
                init_audio_type_radio,
                inversion_steps_slider,
                inversion_gamma_slider,
                inversion_unconditional_checkbox,
                adapt_duration_checkbox,
                duration_padding_slider,
                use_effective_length_checkbox,
                batch_size_state,
                dist_shift_state,
            ] + lora_ui_inputs

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False,
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False))
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])

            if has_inpainting:
                send_to_inpaint_button = gr.Button("Send to inpaint audio", scale=1)
                send_to_inpaint_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[inpaint_audio_input])
    
    generate_button.click(fn=generate_cond,
        inputs=inputs,
        outputs=[
            audio_output,
            audio_spectrogram_output
        ],
        api_name="generate")

def create_diffusion_cond_ui(model_config, in_model, in_model_half=True, gradio_title="", default_prompt=None):
    global model, sample_size, sample_rate, model_type, model_half

    model = in_model
    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]
    model_type = model_config["model_type"]

    model_half = in_model_half

    js ="""function run_javascript_on_page_load(){
        const generateBtn = Array.from(document.querySelectorAll('button'))
            .find(btn => btn.innerText.trim() === 'Generate');
        function getAudioOutputPlayer () {
            return [...document.querySelectorAll('label')].find(label => label.textContent.trim() === 'Output audio')?.parentElement.querySelector('audio');
        }
        const infiniteRadio = document.querySelector('#infinite-radio input[type="checkbox"]');
        const autoplay = document.querySelector('#autoplay input[type="checkbox"]');
        const automaticDownload = document.querySelector('#automatic-download input[type="checkbox"]');
        let radioAutoStart = false;
        let listenersSetup = false;
        const setupListeners = () => {
            const audioEl = getAudioOutputPlayer();
            if (!audioEl) return;
            audioEl.addEventListener('loadedmetadata', () => {
                if(automaticDownload.checked){
                    downloadAudio(audioEl);
                }
                if(autoplay.checked || radioAutoStart){
                    audioEl.play();
                    radioAutoStart = false;
                }
                if(infiniteRadio.checked){
                    audioEl.addEventListener('timeupdate', function checkAudioEnd() {
                        // Can set window.headstart (seconds) in the dev console if you want to start generating before the song is over
                        let headstart = 1;
                        if(window.headstart) headstart = window.headstart;
                        if (audioEl.duration - audioEl.currentTime <= headstart) {                            
                            generateBtn.click();
                            radioAutoStart = true;
                            audioEl.removeEventListener('timeupdate', checkAudioEnd);
                        }
                    });
                }
            });
            listenersSetup = true;
        };
        generateBtn.addEventListener('click', () => {
            if(listenersSetup) return;
            const interval = setInterval(() => {
                console.log("...")
                const audioEl = document.querySelector('audio');
                if (audioEl?.src && audioEl.src !== window.location.href) {
                    setupListeners();
                    clearInterval(interval);
                }
            }, 100);
        });
        // Respond to >> button on MacBookPro and on steering wheel during CarPlay
        if ('mediaSession' in navigator) {
            navigator.mediaSession.setActionHandler('nexttrack', () => generateBtn.click());
            navigator.mediaSession.setActionHandler('play', () => getAudioOutputPlayer()?.play());
            navigator.mediaSession.setActionHandler('pause', () => getAudioOutputPlayer()?.pause());
        }
        // Automatic Download
        function downloadAudio(audioEl) {
            const audioSrc = audioEl.src;
            const link = document.createElement('a');
            link.href = audioSrc;
            link.download = audioSrc.substring(audioSrc.lastIndexOf('/') + 1);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
    """

    with gr.Blocks(js=js, theme=gr.themes.Base()) as ui:
        if gradio_title:
            gr.Markdown("### %s" % gradio_title)
        with gr.Tab("Generation"):
            create_sampling_ui(model_config, default_prompt=default_prompt)

        # JavaScript to autoplay audio immediately after generation (if autoplay enabled)
    return ui
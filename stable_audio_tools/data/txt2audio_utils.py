import taglib
import os
from datetime import datetime
import platform
import subprocess

def open_outputs_path():
    outputs = f"outputs/{datetime.now().strftime('%Y-%m-%d')}"
    if not os.path.isdir(outputs):
        return
    outputs = os.path.abspath(outputs)
    if platform.system() == "Windows":
        os.startfile(outputs)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", outputs])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        subprocess.Popen(["wsl-open", outputs])
    else:
        subprocess.Popen(["xdg-open", outputs])

def create_output_path(suffix):
    outputs = f"outputs/{datetime.now().strftime('%Y-%m-%d')}"
    count = 0

    if os.path.isdir(outputs):
        counts = [os.path.splitext(file)[0].split('-')[0] for file in os.listdir(outputs) if file.endswith(".wav")]
        count = max([int(i) for i in counts if i.isnumeric()]) + 1
    else:
        os.makedirs(outputs)

    return f"{outputs}/{'{:05d}'.format(count)}-{suffix}.wav"

def get_generation_data(file):
    with taglib.File(file) as sound:
        if len(sound.tags) != 1:
            return None
                  
        data = sound.tags["TITLE"]

        if len(data) != 12:
            return None 
        if data[0] == "None":
            data[0] = ""       
        if data[1] == "None":
            data[1] = ""
        if data[5] == "None":
            data[5] = 0

        for i in range(2, 8):
            data[i] = int(data[i])

        for i in range(9, 12):
            data[i] = float(data[i])
        
        data[4] = float(data[4])

        return data

def save_generation_data(sound_path, prompt, negative_prompt, seconds_start, seconds_total, steps, preview_every, cfg_scale, seed, sampler_type, sigma_min, sigma_max, cfg_rescale):
    if prompt == "":
        prompt = "None"
    if negative_prompt == "":
        negative_prompt = "None"

    with taglib.File(sound_path, save_on_exit=True) as sound:
        sound.tags["TITLE"] = [
            prompt, 
            negative_prompt, 
            str(seconds_start),
            str(seconds_total), 
            str(steps), 
            str(preview_every), 
            str(cfg_scale), 
            str(seed), 
            str(sampler_type), 
            str(sigma_min),
            str(sigma_max),
            str(cfg_rescale)]

def txt2audio_css():
    return """
    #prompt_options {
        flex-wrap: nowrap;
        height: 40px;
    }
    """

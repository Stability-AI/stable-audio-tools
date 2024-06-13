import taglib
import os
from datetime import datetime
import platform
import subprocess
import json

def set_selected_model(model_name):
    if model_name in [data["name"] for data in get_models_data()]:
        config = get_config()
        config["model_selected"] = model_name   
        with open("config/txt2audio.json", "w") as file:
            json.dump(config, file, indent=4)
            file.write('\n')

def get_config():
    with open("config/txt2audio.json") as file:
        return json.load(file)

def get_models_name():
    return [model["name"] for model in get_models_data()]

def get_models_data():
    models = []
    file_types = ['.ckpt', '.safetensors', '.pth']
    for file in os.listdir("models/"):
        _file = os.path.splitext(file)
        config_path = f"models/{_file[0]}.json"
        if _file[1] in file_types and os.path.isfile(config_path):
            models.append({"name": _file[0], "path": f"models/{file}", "config_path": config_path})
    return models

def open_outputs_path():
    outputs_dir = "outputs/"
    outputs = outputs_dir + datetime.now().strftime('%Y-%m-%d')
    
    if not os.path.isdir(outputs):
        if not os.path.isdir(outputs_dir):
            return
        else:
            outputs = outputs_dir
            
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
    div.svelte-sa48pu>*, div.svelte-sa48pu>.form>* {
        flex: 1 1 0%;
        flex-wrap: wrap;
        min-width: min(40px, 100%);
    }

    #refresh_btn {
        padding: 0px;
    }

    #selected_model_items div.svelte-1sk0pyu div.wrap.svelte-1sk0pyu div.wrap-inner.svelte-1sk0pyu div.secondary-wrap.svelte-1sk0pyu input.border-none.svelte-1sk0pyu {
        margin: 0px;
    }

    #prompt_options {
        flex-wrap: nowrap;
        height: 40px;
    }

    #selected_model_container {
        gap: 3px;
    }
    """

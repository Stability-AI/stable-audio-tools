import json

from .factory import create_model_from_config
from .utils import load_ckpt_state_dict

from huggingface_hub import hf_hub_download

def get_pretrained_model(name: str):
    
    model_config_path = hf_hub_download(name, filename="model_config.json", repo_type='model')

    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    # Try to download the model.safetensors file first, if it doesn't exist, download the model.ckpt file
    try:
        model_ckpt_path = hf_hub_download(name, filename="model.safetensors", repo_type='model')
    except Exception as e:
        model_ckpt_path = hf_hub_download(name, filename="model.ckpt", repo_type='model')

    model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    return model, model_config
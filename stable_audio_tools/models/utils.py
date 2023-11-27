import torch
from safetensors.torch import load_file

def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    
    return state_dict

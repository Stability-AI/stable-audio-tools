#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import typing as tp
import gc

from torch import nn

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            cond_len: int
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.cond_len = cond_len
        self.proj_out = nn.Linear(dim, output_dim)

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    
class TimingConditioner(Conditioner):
    def __init__(self,
                output_dim: int,
                max_seconds: int = 512):
        super().__init__(output_dim, output_dim, 1)

        self.max_seconds = max_seconds
        self.seconds_start_embedder = nn.Embedding(max_seconds + 1, output_dim)
        self.seconds_total_embedder = nn.Embedding(max_seconds + 1, output_dim)

    def forward(self, seconds_starts_totals: tp.List[tp.Tuple[int, int]], device=None) -> tp.Any:
        
        self.seconds_start_embedder.to(device)
        self.seconds_total_embedder.to(device)

        seconds_starts_totals = torch.tensor(seconds_starts_totals).to(device)
        seconds_starts_totals = seconds_starts_totals.clamp(0, self.max_seconds)
        seconds_starts, seconds_totals = seconds_starts_totals.transpose(0, 1)

        seconds_starts_embeds = self.seconds_start_embedder(seconds_starts).unsqueeze(1)
        seconds_totals_embeds = self.seconds_total_embedder(seconds_totals).unsqueeze(1)

        return seconds_starts_embeds, seconds_totals_embeds

class IntConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                min_val: int=0,
                max_val: int=512
                ):
        super().__init__(output_dim, output_dim, 1)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
            
            self.int_embedder.to(device)
    
            ints = torch.tensor(ints).to(device)
            ints = ints.clamp(self.min_val, self.max_val)
    
            int_embeds = self.int_embedder(ints).unsqueeze(1)
    
            return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class CLAPTextConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True):
        super().__init__(512, output_dim, 1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        import laion_clap
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device=device).requires_grad_(False).eval()

        self.model.load_ckpt(clap_ckpt_path)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, texts: tp.List[str], device: tp.Any = None) -> tp.Any:

        self.model.to(device)
        self.proj_out.to(device)

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]


class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, max_length)
        
        from transformers import T5EncoderModel, T5Tokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)
            finally:
                logging.disable(previous_level)
            
        self.model = model


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        self.model.to(device)
        self.proj_out.to(device)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        with torch.set_grad_enabled(self.enable_grad):
            
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]    

        embeddings = self.proj_out(embeddings)

        return [embeddings, attention_mask]


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner]):
        super().__init__()

        self.conditioners = conditioners

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            if key in batch_metadata[0]:
                conditioner_inputs = [x[key][0] if isinstance(x[key], list) or isinstance(x[key], tuple) else x[key] for x in batch_metadata]
                
                output[key] = conditioner(conditioner_inputs, device)

        return output
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]

    for conditioner_config in config["configs"]:
        id = conditioner_config["id"]

        conditioner_type = conditioner_config["type"]

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(output_dim=cond_dim, **conditioner_config["config"])
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(output_dim=cond_dim, **conditioner_config["config"])
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(output_dim=cond_dim, **conditioner_config["config"])
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners)
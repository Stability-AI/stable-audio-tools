#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import string
import typing as tp
import gc
from enum import Enum
import os


class PaddingMode(str, Enum):
    """Enum for handling padding in text conditioner embeddings."""
    NONE = "none"       # No padding handling (raw embeddings with pad token)
    ZERO = "zero"       # Zero out padding positions (default)
    LEARNED = "learned" # Use learned padding embedding

from .adp import NumberEmbedder
from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..models.utils import copy_state_dict
from .utils import load_ckpt_state_dict, enable_torch_compile
from .transformer import AbsolutePositionalEmbedding

from torch import nn
from typing import Union, Dict, List, Tuple
from torch.nn import functional as F
import numpy as np

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False,
            padding_mode: str = "zero"
            ):

        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.padding_mode = padding_mode
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

        # Learned padding embedding (only created if needed)
        if padding_mode == "learned" or padding_mode == PaddingMode.LEARNED:
            self.padding_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

    def apply_padding(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply padding handling based on padding_mode.

        Args:
            embeddings: [batch, seq_len, dim] - the embeddings to process
            attention_mask: [batch, seq_len] bool/int, True/1 = valid token

        Returns:
            embeddings with padding handled according to mode
        """
        mode = self.padding_mode
        if isinstance(mode, str):
            mode = PaddingMode(mode)

        if mode == PaddingMode.NONE:
            return embeddings
        elif mode == PaddingMode.ZERO:
            return embeddings * attention_mask.unsqueeze(-1).float()
        elif mode == PaddingMode.LEARNED:
            mask_expanded = attention_mask.unsqueeze(-1).bool()
            return torch.where(
                mask_expanded,
                embeddings,
                self.padding_embedding.unsqueeze(0).unsqueeze(0).expand_as(embeddings)
            )
        else:
            raise ValueError(f"Unknown padding mode: {mode}")

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()

class IntConditioner(Conditioner):
    def __init__(self,
                output_dim: int,
                min_val: int=0,
                max_val: int=512
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:

            #self.int_embedder.to(device)

            ints = torch.tensor(ints).to(device)
            ints = ints.clamp(self.min_val, self.max_val)

            int_embeds = self.int_embedder(ints).unsqueeze(1)

            return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self,
                output_dim: int,
                min_val: float=0,
                max_val: float=1,
                fourier_features_type : tp.Literal["learned", "expo"] = "learned"
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim, fourier_features_type=fourier_features_type)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
            self.embedder.to(device)
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)

            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)

            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]

class ListConditioner(Conditioner):
    def __init__(self,
                output_dim: int,
                options: tp.List[str]
                ):
        super().__init__(output_dim, output_dim)

        self.options = options
        self.embedder = nn.Embedding(len(options)+1, output_dim).requires_grad_(True)

    def forward(self, texts: tp.List[str], device=None) -> tp.Any:
        self.embedder.to(device)
        # Cast the inputs to floats, handling the case where the input is not in the options
        ints = [self.options.index(x) + 1 if x in self.options else 0 for x in texts]

        ints = torch.tensor(ints).to(device) # shape [batch_size]

        int_embeds = self.embedder(ints).unsqueeze(1) # shape [batch_size, 1, output_dim]

        return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class SATCLAPTextConditioner(Conditioner):
    def __init__(self,
                clap_model,
                output_dim: int,
                project_out: bool = False,
                use_text_features = False,
                feature_layer_ix: int = -2,
                **kwargs):

        super().__init__(clap_model.text_branch.embed_dim, output_dim, project_out=project_out)

        self.model = clap_model
        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix

        self.model.requires_grad_(False)
        self.model.eval()

        del self.model.pretransform
        del self.model.audio_branch

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.model.text_branch.get_text_features([texts[0], ""], layer_ix=self.feature_layer_ix)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.model.text_branch.get_text_features(texts, layer_ix=self.feature_layer_ix)

            # Cast text feature to same type as proj_out, unless proj_out is Identity
            if not isinstance(self.proj_out, nn.Identity):
                proj_out_dtype = next(self.proj_out.parameters()).dtype
                text_features = text_features.to(proj_out_dtype)

            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""])[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        # Cast text embedding to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            text_embedding = text_embedding.to(proj_out_dtype)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class SATCLAPAudioConditioner(Conditioner):
    def __init__(self,
                clap_model,
                output_dim: int,
                project_out: bool = False,
                **kwargs):

        super().__init__(clap_model.joint_embed_dim, output_dim, project_out=project_out)

        self.model = clap_model

        self.model.requires_grad_(False)
        self.model.eval()

        del self.model.text_branch

    def forward(self, latents: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if isinstance(latents, list) or isinstance(latents, tuple):
            latents = torch.stack(latents, dim=0)

        latents = latents.to(device)

        audio_embedding = self.model.get_audio_embedding(latents)

        # Cast text embedding to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            audio_embedding = audio_embedding.to(proj_out_dtype)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]

def clap_load_state_dict(clap_ckpt_path, clap_model):
    state_dict = torch.load(clap_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    # Remove "module." from state dict keys
    state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Fix for transformers library
    removed_keys = ["text_branch.embeddings.position_ids"]
    for removed_key in removed_keys:
        if removed_key in state_dict:
            del state_dict[removed_key]

    clap_model.load_state_dict(state_dict)

class CLAPTextConditioner(Conditioner):
    def __init__(self,
                 output_dim: int,
                 clap_ckpt_path,
                 use_text_features = False,
                 feature_layer_ix: int = -1,
                 audio_model_type="HTSAT-base",
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False,
                 padding_mode: str = "none"):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out, padding_mode=padding_mode)

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap

                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else:
                    self.__dict__["model"] = model

                clap_load_state_dict(clap_ckpt_path, self.model.model)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2, device: tp.Any = "cuda"):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix, device=device)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix, device=device)

            # Cast text feature to same type as proj_out, unless proj_out is Identity
            if not isinstance(self.proj_out, nn.Identity):
                proj_out_dtype = next(self.proj_out.parameters()).dtype
                text_features = text_features.to(proj_out_dtype)

            text_features = self.proj_out(text_features)
            text_features = self.apply_padding(text_features, text_attention_mask)

            return [text_features, text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        # Cast text embedding to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            text_embedding = text_embedding.to(proj_out_dtype)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class CLAPAudioConditioner(Conditioner):
    def __init__(self,
                 output_dim: int,
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base",
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap

                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else:
                    self.__dict__["model"] = model

                clap_load_state_dict(clap_ckpt_path, self.model.model)

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Any = "cuda") -> tp.Any:

        self.model.to(device)

        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1)

        with torch.amp.autocast('cuda', enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        # Cast audio embedding to same type as proj_out, unless proj_out is Identity

        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            audio_embedding = audio_embedding.to(proj_out_dtype)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]

class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl", "google/t5-v1_1-xl", "google/t5-v1_1-xxl"]

    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/t5-v1_1-xl": 2048,
        "google/t5-v1_1-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            padding_mode: str = "zero",
            model_path: str = None,
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out, padding_mode=padding_mode)

        load_from = model_path or t5_model_name

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Set environment variables to disable progress bars BEFORE importing transformers
        prev_hf_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        prev_transformers = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import T5EncoderModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(load_from)
                model = T5EncoderModel.from_pretrained(load_from).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)

            finally:
                logging.disable(previous_level)
                # Restore environment variables
                if prev_hf_hub is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_hub
                if prev_transformers is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_transformers

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.model.to(device)
        self.proj_out.to(device)

        if isinstance(texts[0], dict):
            # Pre-tokenized input (e.g. from DataLoader with tokenizers)
            input_ids = torch.stack([x["input_ids"] for x in texts]).to(device, non_blocking=True)
            attention_mask = torch.stack([x["attention_mask"] for x in texts]).to(device, non_blocking=True).to(torch.bool)
        else:
            encoded = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()

        with torch.amp.autocast('cuda', dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]

        # Cast embeddings to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.proj_out(embeddings)
        embeddings = self.apply_padding(embeddings, attention_mask)

        return embeddings, attention_mask

class T5GemmaConditioner(Conditioner):

    T5GEMMA_MODELS = ["google/t5gemma-b-b-ul2"]

    T5GEMMA_MODEL_DIMS = {
        "google/t5gemma-b-b-ul2": 768,
    }

    def __init__(
            self,
            output_dim: int,
            model_name: str = "google/t5gemma-b-b-ul2",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            padding_mode: str = "zero",
            model_path: str = None,
            repo_id: str = None,
            subfolder: str = None,
    ):
        assert model_name in self.T5GEMMA_MODELS, f"Unknown T5 model name: {model_name}"
        super().__init__(self.T5GEMMA_MODEL_DIMS[model_name], output_dim, project_out=project_out, padding_mode=padding_mode)

        load_from = model_path or repo_id or model_name

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Set environment variables to disable progress bars BEFORE importing transformers
        # This is the most reliable way to suppress HuggingFace progress bars
        prev_hf_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        prev_transformers = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import T5GemmaEncoderModel, AutoTokenizer, AutoConfig
                logging.info(f"Loading T5Gemma tokenizer and model from: {load_from}")
                hf_kwargs = {"subfolder": subfolder} if subfolder else {}
                self.tokenizer = AutoTokenizer.from_pretrained(load_from, **hf_kwargs)
                config = AutoConfig.from_pretrained(load_from, **hf_kwargs)
                config.is_encoder_decoder = False
                model = T5GemmaEncoderModel.from_pretrained(load_from, config=config, **hf_kwargs).train(enable_grad).requires_grad_(enable_grad)

            finally:
                logging.disable(previous_level)
                # Restore environment variables
                if prev_hf_hub is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_hub
                if prev_transformers is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_transformers

        # Compile the model to reduce CPU-GPU kernel launch overhead,
        # which is sensitive to CPU contention from DataLoader workers
        if enable_torch_compile:
            model = torch.compile(model)

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

        self._device_initialized = False

    def forward(self, inputs: tp.Union[tp.List[str], tp.List[tp.Dict[str, torch.Tensor]]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        # Only move to device once (avoid overhead on every forward call)
        if not self._device_initialized:
            self.model.to(device)
            self.proj_out.to(device)
            self.model.eval()
            self._device_initialized = True

        # Handle pre-tokenized inputs (dicts with input_ids/attention_mask from DataLoader workers)
        # or raw strings (from demo generation / inference)
        if isinstance(inputs[0], dict):
            input_ids = torch.stack([x["input_ids"] for x in inputs]).to(device, non_blocking=True)
            attention_mask = torch.stack([x["attention_mask"] for x in inputs]).to(device, non_blocking=True).to(torch.bool)
        else:
            encoded = self.tokenizer(
                inputs,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True).to(torch.bool)

        with torch.no_grad():
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]

        # Cast embeddings to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.proj_out(embeddings)
        embeddings = self.apply_padding(embeddings, attention_mask)

        return embeddings, attention_mask

class CausalLMConditioner(Conditioner):

    MODELS = ["google/gemma-2-2b"]

    MODEL_DIMS = {
        "google/gemma-2-2b": 2304
    }

    def __init__(
            self,
            output_dim: int,
            model_name: str = "google/gemma-2-2b",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            learned_scale: bool = True,
            padding_mode: str = "zero",
            model_path: str = None,
    ):
        assert model_name in self.MODELS, f"Unknown model name: {model_name}"
        super().__init__(self.MODEL_DIMS[model_name], output_dim, project_out=project_out, padding_mode=padding_mode)

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from .blocks import RMSNorm

        load_from = model_path or model_name

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Set environment variables to disable progress bars BEFORE importing transformers
        prev_hf_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        prev_transformers = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.tokenizer = AutoTokenizer.from_pretrained(load_from)
                model = AutoModelForCausalLM.from_pretrained(load_from).train(enable_grad).requires_grad_(enable_grad)

            finally:
                logging.disable(previous_level)
                # Restore environment variables
                if prev_hf_hub is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_hub
                if prev_transformers is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_transformers

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

        self.norm = RMSNorm(self.dim)

        self.learned_scale = learned_scale

        if self.learned_scale:
            self.scale = nn.Parameter(torch.tensor(.01))


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()

        with torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
            )["hidden_states"][-1]

        # Cast embeddings to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.norm(embeddings)

        if self.learned_scale:
            embeddings = embeddings * self.scale

        embeddings = self.proj_out(embeddings)
        embeddings = self.apply_padding(embeddings, attention_mask)

        return embeddings, attention_mask

class PhonemeConditioner(Conditioner):
    """
    A conditioner that turns text into phonemes and embeds them using a lookup table
    Only works for English text

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)

        from g2p_en import G2p

        self.max_length = max_length

        self.g2p = G2p()

        # Reserving 0 for padding, 1 for ignored
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [self.g2p(text) for text in texts] # shape [batch_size, length]

        phoneme_ignore = [" ", *string.punctuation]

        # Remove ignored phonemes and cut to max length
        batch_phonemes = [[p if p not in phoneme_ignore else "_" for p in phonemes] for phonemes in batch_phonemes]

        # Convert to ids
        phoneme_ids = [[self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes] for phonemes in batch_phonemes]

        #Pad to match longest and make a mask tensor for the padding
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]

        phoneme_ids = torch.tensor(phoneme_ids).to(device)

        # Convert to embeddings
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)

        phoneme_embeds = self.proj_out(phoneme_embeds)

        return phoneme_embeds, torch.ones(phoneme_embeds.shape[0], phoneme_embeds.shape[1]).to(device)

class TokenizerLUTConditioner(Conditioner):
    """
    A conditioner that embeds text using a lookup table on a pretrained tokenizer's vocabulary

    Args:
        tokenizer_name: the name of the tokenizer from the Hugging Face transformers library
        output_dim: the dimension of the output embeddings
        max_length: the maximum length of the text to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            tokenizer_name: str, # Name of a tokenizer from the Hugging Face transformers library
            output_dim: int,
            max_length: int = 1024,
            use_abs_pos_emb = False,
            project_out: bool = False,
            special_tokens: tp.List[str] = [],
            model_path: str = None,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)

        from transformers import AutoTokenizer

        load_from = model_path or tokenizer_name

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        # Also suppress transformers-specific logging and progress bars
        transformers_log_level = None
        transformers_disable_progress_bar = None
        try:
            import transformers
            transformers_log_level = transformers.logging.get_verbosity()
            transformers.logging.set_verbosity_error()
            # Disable progress bars
            try:
                from transformers.utils import is_progress_bar_enabled
                transformers_disable_progress_bar = not is_progress_bar_enabled()
                transformers.utils.logging.disable_progress_bar()
            except (ImportError, AttributeError) as e:
                # Progress bar control not available in this transformers version
                logging.debug(f"Could not disable transformers progress bar: {e}")
        except (ImportError, AttributeError) as e:
            # Transformers not available or version mismatch
            logging.debug(f"Could not configure transformers logging: {e}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(load_from)

            finally:
                logging.disable(previous_level)
                if transformers_log_level is not None:
                    try:
                        transformers.logging.set_verbosity(transformers_log_level)
                    except (AttributeError, Exception) as e:
                        logging.debug(f"Could not restore transformers log level: {e}")
                if transformers_disable_progress_bar is not None and not transformers_disable_progress_bar:
                    try:
                        transformers.utils.logging.enable_progress_bar()
                    except (AttributeError, Exception) as e:
                        logging.debug(f"Could not re-enable transformers progress bar: {e}")

        # Add special tokens
        if len(special_tokens) > 0:
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        self.max_length = max_length

        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

        self.abs_pos_emb = None

        if use_abs_pos_emb:
            self.abs_pos_emb = AbsolutePositionalEmbedding(output_dim, max_length)

    def forward(self, inputs: tp.Union[tp.List[str], tp.List[tp.Dict[str, torch.Tensor]]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)

        # Handle pre-tokenized inputs (dicts with input_ids/attention_mask from DataLoader workers)
        # or raw strings (from demo generation / inference)
        if isinstance(inputs[0], dict):
            input_ids = torch.stack([x["input_ids"] for x in inputs]).to(device, non_blocking=True)
            attention_mask = torch.stack([x["attention_mask"] for x in inputs]).to(device, non_blocking=True).to(torch.bool)
        else:
            encoded = self.tokenizer(
                inputs,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True).to(torch.bool)

        embeddings = self.token_embedder(input_ids)

        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        if self.abs_pos_emb is not None:
            embeddings = embeddings + self.abs_pos_emb(embeddings)

        return embeddings, attention_mask

class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int, save_pretransform: bool = False):
        super().__init__(pretransform.encoded_channels, output_dim)


        if not save_pretransform:
            self.__dict__["pretransform"] = pretransform
        else:
            self.pretransform = pretransform


    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.stack(audio, dim=0)

        # Add batch dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)

        audio = audio.to(device)

        latents = self.pretransform.encode(audio)

        latents = self.proj_out(latents)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]

class SourceMixConditioner(Conditioner):
    """
    A conditioner that mixes projected audio embeddings from multiple sources

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
        source_keys: a list of keys for the potential sources in the metadata

    """
    def __init__(
        self,
        pretransform: Pretransform,
        output_dim: int,
        save_pretransform: bool = False,
        source_keys: tp.List[str] = [],
        pre_encoded: bool = False,
        allow_null_source=False,
        source_length=None
    ):
        super().__init__(pretransform.encoded_channels, output_dim)

        if not save_pretransform:
            self.__dict__["pretransform"] = pretransform
        else:
            self.pretransform = pretransform

        self.source_keys = source_keys

        self.source_heads = nn.ModuleList([nn.Conv1d(pretransform.encoded_channels, output_dim, kernel_size=1) for _ in source_keys])

        self.pre_encoded = pre_encoded

        self.allow_null_source = allow_null_source

        if self.allow_null_source:
            self.null_source = nn.Parameter(torch.randn(output_dim, 1))

            assert source_length is not None, "Source length must be specified if allowing null sources"

            self.source_length = source_length

    def forward(self, sources: tp.List[tp.Dict[str, torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        dtype = next(self.proj_out.parameters()).dtype

        # Output has to be the batch of summed projections
        # Input is per-batch-item list of source audio

        mixes = []

        for source_dict in sources: # Iterate over batch items

            mix = None

            for key_ix, key in enumerate(self.source_keys): # Iterate over potential sources
                if key in source_dict:

                    source = source_dict[key]

                    if not self.pre_encoded:
                        assert source.dim() == 2, f"Source audio must be shape [channels, samples], got shape: {source.shape}"
                        audio = set_audio_channels(source.unsqueeze(0), self.pretransform.io_channels)

                        audio = audio.to(device)
                        latents = self.pretransform.encode(audio).squeeze(0)
                    else:
                        latents = source.to(device)

                    latents = latents.to(dtype)

                    if mix is None:
                        mix = self.source_heads[key_ix](latents)
                    else:
                        mix += self.source_heads[key_ix](latents)

            if mix is not None:
                mixes.append(mix)
            else:
                if self.allow_null_source:
                    mixes.append(self.null_source.repeat(1, self.source_length))
                else:
                    raise ValueError("No sources found for mix")

        mixes = torch.stack(mixes, dim=0)

        return [mixes, torch.ones(mixes.shape[0], mixes.shape[2]).to(mixes.device)]


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}, pre_encoded_keys: tp.List[str] = []):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys
        self.pre_encoded_keys = pre_encoded_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]

                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)

            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device)

        return output

def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any], pretransform=None) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]

    default_keys = config.get("default_keys", {})

    pre_encoded_keys = config.get("pre_encoded_keys", [])

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}

        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "t5gemma":
            conditioners[id] = T5GemmaConditioner(**conditioner_config)
        elif conditioner_type == "causal_lm":
            conditioners[id] = CausalLMConditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "list":
            conditioners[id] = ListConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "sat_clap_text":
            from .clap import create_clap_from_config

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            clap_model = create_clap_from_config(conditioner_config, pretransform=pretransform if use_model_pretransform else None)

            clap_ckpt_path = conditioner_config.get("ckpt_path", None)

            if clap_ckpt_path is not None:
                copy_state_dict(clap_model, load_ckpt_state_dict(clap_ckpt_path))

                # Ensure that loading the checkpoint doesn't overwrite the model's pretransform
                if use_model_pretransform:
                    clap_model.pretransform = pretransform

            conditioners[id] = SATCLAPTextConditioner(clap_model, **conditioner_config)

        elif conditioner_type == "sat_clap_audio":
            from .clap import create_clap_from_config

            sample_rate = conditioner_config.get("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for SAT-CLAP conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            clap_model = create_clap_from_config(conditioner_config, pretransform=pretransform if use_model_pretransform else None)

            clap_ckpt_path = conditioner_config.get("ckpt_path", None)

            if clap_ckpt_path is not None:
                copy_state_dict(clap_model, load_ckpt_state_dict(clap_ckpt_path))

                # Ensure that loading the checkpoint doesn't overwrite the model's pretransform
                if use_model_pretransform:
                    clap_model.pretransform = pretransform

            conditioners[id] = SATCLAPAudioConditioner(clap_model, **conditioner_config)

        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for pretransform conditioners"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(cond_pretransform, **conditioner_config)
        elif conditioner_type == "source_mix":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for source_mix conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for source_mix conditioners if use_model_pretransform is True"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = SourceMixConditioner(cond_pretransform, **conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)
"""
Adapted from https://github.com/LAION-AI/CLAP under CC0-1.0 license
License available at LICENSES/LICENSE_CLAP.txt
"""

import logging, warnings
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .blocks import FourierFeatures
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from .transformer import ContinuousTransformer

from transformers import RobertaModel, RobertaTokenizer

class CLAPAudioBranch(nn.Module):
    def __init__(self, embed_dim, max_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

    def forward(self, audio):
        raise NotImplementedError

class CLAPTextBranch(nn.Module):
    def __init__(self, embed_dim, max_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

    def forward(self, text):
        raise NotImplementedError

class LatentTransformerCLAPAudioBranch(CLAPAudioBranch):
    def __init__(
        self, 
        embed_dim: int,
        embed_type: tp.Literal["cls_prepend", "avg_pool"] = "cls_prepend",
        noise_config = None,
        max_length: int = 1024,
        **transformer_kwargs
        ):
        super().__init__(embed_dim, max_length)

        self.noise_config = noise_config
        if self.noise_config is not None:
            noise_features_dim=256
            noise_embed_dim = noise_config.get("embed_dim", 64)
            self.noise_features = FourierFeatures(1, 256)
            self.to_noise_embed = nn.Linear(256, noise_embed_dim)

            transformer_kwargs["global_cond_dim"] = noise_embed_dim

        self.transformer = ContinuousTransformer(dim=embed_dim, **transformer_kwargs)

        self.embed_type = embed_type

        if self.embed_type == "cls_prepend":
            self.cls_embedding = nn.Parameter(torch.randn(1, 1, self.transformer.dim))
        elif self.embed_type == "avg_pool":
            self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.embed_dim = self.transformer.dim

    def forward(self, latents, mask=None, noise_levels=None, return_hidden_states=False, **kwargs):

        if self.noise_config is not None:
            if noise_levels is None:
                # Assume latents aren't noised, set noise levels to 0
                noise_levels = torch.zeros(latents.shape[0], device=latents.device)

            noise_embed = self.to_noise_embed(self.noise_features(noise_levels[:, None])) # [B, noise_embed_dim]
        else:
            noise_embed = None

        latents = latents.permute(0, 2, 1)

        # Wire mask → padding_mask for transformer varlen attention masking
        # CLAP training passes mask=padding_masks; ARC passes padding_mask= via **kwargs
        if mask is not None and 'padding_mask' not in kwargs:
            kwargs['padding_mask'] = mask

        if self.embed_type == "cls_prepend":

            cls_embedding = self.cls_embedding.expand(latents.shape[0], -1, -1)

            output, info = self.transformer(latents, prepend_embeds=cls_embedding, return_info=True, global_cond=noise_embed, **kwargs)

            embedding = output[:, 0, :]
        elif self.embed_type == "avg_pool":
            output, info = self.transformer(latents, return_info=True, global_cond=noise_embed, **kwargs)

            padding_mask = kwargs.get('padding_mask', None)
            if padding_mask is not None:
                # Strip any prepended memory tokens — they precede the sequence in output
                num_memory = getattr(self.transformer, 'num_memory_tokens', 0)
                output_seq = output[:, num_memory:, :] if num_memory > 0 else output
                # Masked mean pool: only average over valid (non-padded) positions
                mask_float = padding_mask.unsqueeze(-1).float()  # [B, T, 1]
                embedding = (output_seq * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
            else:
                embedding = self.avg_pool(output.permute(0, 2, 1)).squeeze(-1)

        output_dict = {
            "embedding": embedding
        }

        if return_hidden_states:
            output_dict["hidden_states"] = info["hidden_states"]

        return output_dict

class RobertaCLAPTextBranch(CLAPTextBranch):

    ROBERTA_MODEL_DIMS = {
        "roberta-base": 768,
    }

    def __init__(
        self, 
        name: str,
        max_length: int,
    ):
        super().__init__(self.ROBERTA_MODEL_DIMS[name], max_length)
        self.tokenizer = RobertaTokenizer.from_pretrained(name)
        self.model = RobertaModel.from_pretrained(name)

    def get_text_features(self, text, layer_ix=-2):

        device = next(self.model.parameters()).device

        text = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.max_length, 
            truncation=True
        )

        attention_mask = text["attention_mask"].to(device=device, non_blocking=True)
        text_features= self.model(
            input_ids=text["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return text_features, attention_mask

    def forward(self, text):

        device = next(self.model.parameters()).device

        text = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.max_length, 
            truncation=True
        )

        embedding = self.model(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
            )["pooler_output"]

        output_dict = {
            "embedding": embedding
        }

        return output_dict

class CLAP(nn.Module):
    def __init__(
        self,
        joint_embed_dim: int,
        audio_branch: nn.Module,
        text_branch: nn.Module,
        pretransform: Pretransform,
        model_objective: tp.Literal["clap", "slap"] = "clap"
    ):
        super().__init__()
        self.joint_embed_dim = joint_embed_dim

        self.audio_branch = audio_branch

        self.text_branch = text_branch

        self.pretransform = pretransform
        
        audio_embed_dim = self.audio_branch.embed_dim
        
        text_embed_dim = self.text_branch.embed_dim

        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, self.joint_embed_dim),
            nn.ReLU(),
            nn.Linear(self.joint_embed_dim, self.joint_embed_dim)
        )

        self.audio_projection = nn.Sequential(
                nn.Linear(audio_embed_dim, self.joint_embed_dim),
                nn.ReLU(),
                nn.Linear(self.joint_embed_dim, self.joint_embed_dim)
            )

        self.model_objective = model_objective

        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        nn.init.constant_(self.logit_scale_a, np.log(1 / 0.07))

    def get_text_embedding(self, text):
        text_embedding = self.text_projection(self.text_branch(text)["embedding"])

        return F.normalize(text_embedding, dim=-1)

    def get_audio_embedding(self, audio_features, **audio_kwargs):
        audio_embedding = self.audio_projection(self.audio_branch(audio_features, **audio_kwargs)["embedding"])

        return F.normalize(audio_embedding, dim=-1)

    def get_audio_features(self, audio):
        return self.pretransform.encode(audio)

def create_clap_audio_branch_from_config(model_config: dict):

    audio_branch_config = model_config["model"]["audio"]

    audio_branch_model_info = audio_branch_config["model"]

    audio_context_length = audio_branch_config.get("context_length", 1024)

    audio_branch_model_type = audio_branch_model_info["type"]

    audio_branch_model_config = audio_branch_model_info["config"]

    if audio_branch_model_type == "transformer":
        audio_branch = LatentTransformerCLAPAudioBranch(max_length=audio_context_length, **audio_branch_model_config)
    else:
        raise ValueError(f"Unknown audio branch type: {audio_branch_model_type}")

    return audio_branch

def create_clap_text_branch_from_config(model_config: dict):

    text_branch_config = model_config["model"]["text"]

    text_context_length = text_branch_config.get("context_length", 77)

    text_branch_model_info = text_branch_config["model"]
    text_branch_type = text_branch_model_info["type"]

    text_branch_model_config = text_branch_model_info["config"]

    if text_branch_type == "roberta":
        text_branch = RobertaCLAPTextBranch(max_length=text_context_length, **text_branch_model_config)
    else:
        raise ValueError(f"Unknown text branch type: {text_branch_type}")

    return text_branch

def create_clap_from_config(model_config: dict, pretransform=None):
    clap_model_config = model_config["model"]

    joint_embed_dim = clap_model_config["joint_embed_dim"]

    audio_branch = create_clap_audio_branch_from_config(model_config)
    text_branch = create_clap_text_branch_from_config(model_config)

    model_objective = model_config["model"].get("model_objective", "clap")

    if pretransform is None and "pretransform" in clap_model_config:
        pretransform_config = model_config["model"]["pretransform"]
        pretransform = create_pretransform_from_config(pretransform_config, model_config["sample_rate"])

    model = CLAP(
        joint_embed_dim=joint_embed_dim,
        audio_branch=audio_branch,
        text_branch=text_branch,
        pretransform=pretransform,
        model_objective=model_objective)

    return model
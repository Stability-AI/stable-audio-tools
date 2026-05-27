"""
Adapted from https://github.com/LAION-AI/CLAP under CC0-1.0 license
License available at LICENSES/LICENSE_CLAP.txt
"""

import pytorch_lightning as pl
import sys, gc
import numpy as np
import random
import torch
import torchaudio
import typing as tp
import wandb

from einops import rearrange
from safetensors.torch import save_file
from torch import optim, nn
from torch.nn import functional as F
from tqdm import tqdm
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..models.clap import CLAP

from ..inference.sampling import truncated_logistic_normal_rescaled

from inf_cl import cal_inf_loss

from .utils import create_optimizer_from_config, create_scheduler_from_config, resize_padding_mask, StaggeredLogger

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def cross_uniform_loss(x, y, t=2):
    return torch.cdist(x, y, p=2).pow(2).mul(-t).exp().mean().log()

def align_loss(x, y, alpha=2):
    return (1 - F.cosine_similarity(x, y, dim=-1)).mean()

def chunked_cross_entropy(hidden_states, labels, lm_head, num_chunks=8):
    """Compute causal LM cross-entropy loss without materializing full (B, T, vocab) logits.

    Applies the lm_head in chunks along the sequence dimension and accumulates
    the CE loss, keeping peak memory at (B, chunk_size, vocab) instead of (B, T, vocab).
    Handles the causal shift (predict token i+1 from hidden state i) internally.
    """
    shifted_hidden = hidden_states[:, :-1, :]
    shifted_labels = labels[:, 1:]

    seq_len = shifted_hidden.shape[1]
    chunk_size = (seq_len + num_chunks - 1) // num_chunks
    total_loss_sum = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
    total_valid_tokens = 0

    for i in range(0, seq_len, chunk_size):
        chunk_h = shifted_hidden[:, i:i+chunk_size]
        chunk_l = shifted_labels[:, i:i+chunk_size]
        chunk_logits = lm_head(chunk_h)
        valid = (chunk_l != -100).sum()
        if valid > 0:
            total_loss_sum = total_loss_sum + F.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.shape[-1]),
                chunk_l.reshape(-1),
                ignore_index=-100,
                reduction='sum'
            )
            total_valid_tokens += valid

    return total_loss_sum / max(total_valid_tokens, 1)

# Distributed functions from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
# License can be found at LICENSES/LICENSE_OPEN_CLIP.txt

def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv

def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)

class CLAPTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: CLAP,
            lr: float = None,
            optimizer_configs: dict = None,
            pre_encoded: bool = False,
            htsat_dataset = False,
            loss_config: dict = None,
            mask_padding_attention: bool = False
    ):
        super().__init__()

        self.model = model

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "model": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }

        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        self.loss_config = {
            "contrastive": 1.0,
            "uniformity": 0.1,
            "cross_uniformity": 0.1,
            "alignment": 0.1
        }

        if loss_config is not None:    
            self.loss_config.update(loss_config)
        
        self.pre_encoded = pre_encoded
        self.htsat_dataset = htsat_dataset

        self.all_val_text_features = []
        self.all_val_audio_features = []

        self.audio_noise_config = self.model.audio_branch.noise_config
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        if self.audio_noise_config is not None:
            self.interp_type = self.audio_noise_config.get("interp_type", "linear")

        self.mask_padding_attention = mask_padding_attention

        self._staggered_logger = StaggeredLogger(every_n_steps=10)

    def configure_optimizers(self):
        opt_clap_config = self.optimizer_configs['model']

        if opt_clap_config['optimizer'].get('type') == 'MuonAdamW':
            opt_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        else:
            opt_params = list(self.model.parameters())

        optimizer_clap = create_optimizer_from_config(opt_clap_config['optimizer'], opt_params)

        optimizers = [optimizer_clap]
        schedulers = []

        if "scheduler" in opt_clap_config:
            sched_diff = create_scheduler_from_config(opt_clap_config['scheduler'], optimizer_clap)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            schedulers.append(sched_diff_config)

        return optimizers, schedulers

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, audio_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * audio_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def training_step(self, batch, batch_idx):
        audios, metadata = batch

        if self.htsat_dataset:
            audio_features = self.model.get_audio_embedding(audios, preprocessed=True)
            prompts = [md["prompt"] for md in metadata]
        else:
            if audios.ndim == 4 and audios.shape[0] == 1:
                audios = audios[0]
                prompts = [md["prompt"][0] for md in metadata]
                padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
            else:
                prompts = [md["prompt"] for md in metadata]
                padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
            
            pretransform = self.model.pretransform

            if pretransform is not None:
                pretransform.to(self.device)

                if not self.pre_encoded:
                    with torch.amp.autocast("cuda"):
                        audios = pretransform.encode(audios)

                        padding_masks = resize_padding_mask(padding_masks, audios.shape[2])
                else:
                    if padding_masks.shape[-1] != audios.shape[-1]:
                        padding_masks = resize_padding_mask(padding_masks, audios.shape[-1])

            attention_mask = padding_masks if self.mask_padding_attention else None

            if self.audio_noise_config is not None:

                # Uniformly sample noise levels across the batch
                noise_levels = self.rng.draw(audios.shape[0])[:, 0].to(self.device)

                # set 50% of the noise levels to 0
                p_no_noise = 0.5

                noise_levels = torch.where(torch.rand_like(noise_levels) < p_no_noise, torch.zeros_like(noise_levels), noise_levels)

                if self.interp_type == "linear":
                    alphas, sigmas = 1-noise_levels, noise_levels
                    alphas = alphas[:, None, None]
                    sigmas = sigmas[:, None, None]
                    noise = torch.randn_like(audios)
                    noised_audios = alphas * audios + sigmas * noise

                audio_features = checkpoint(self.model.get_audio_embedding, noised_audios, mask=attention_mask, noise_levels = noise_levels)
            else:
                audio_features = checkpoint(self.model.get_audio_embedding, audios, mask=attention_mask)
        
        text_features = self.model.get_text_embedding(prompts)

        if self.loss_config["contrastive"] > 0:
            t = torch.exp(self.model.logit_scale_a)

            a2t_loss = checkpoint(cal_inf_loss, audio_features, text_features, scale=t)
            t2a_loss = checkpoint(cal_inf_loss, text_features, audio_features, scale=t)

            contrastive_loss = ((a2t_loss + t2a_loss) / 2).mean()

            contrastive_loss_show = self.all_gather(contrastive_loss.detach().clone()).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)
            contrastive_loss_show = contrastive_loss.detach()

        if self.loss_config["alignment"] > 0:
            alignment_loss = checkpoint(align_loss, audio_features, text_features) * self.loss_config["alignment"]
        else:
            alignment_loss = torch.tensor(0.0, device=self.device)

        if self.loss_config["uniformity"] > 0:
            audio_uniformity = checkpoint(uniform_loss, audio_features)
            text_uniformity = checkpoint(uniform_loss, text_features)

            uniformity = ((audio_uniformity + text_uniformity) / 2) * self.loss_config["uniformity"]
        else:
            uniformity = torch.tensor(0.0, device=self.device)
            audio_uniformity = torch.tensor(0.0, device=self.device)
            text_uniformity = torch.tensor(0.0, device=self.device)
            
        if self.loss_config["cross_uniformity"] > 0:
            cross_uniformity = checkpoint(cross_uniform_loss, audio_features, text_features) * self.loss_config["cross_uniformity"]
        else:
            cross_uniformity = torch.tensor(0.0, device=self.device)


        total_loss = contrastive_loss + alignment_loss + uniformity + cross_uniformity

        log_dict = {
            'train/contrastive_loss': contrastive_loss_show,
            'train/alignment_loss': alignment_loss.detach(),
            'train/audio_uniformity': audio_uniformity.detach(),
            'train/text_uniformity': text_uniformity.detach(),
            'train/uniformity': uniformity.detach(),
            'train/cross_uniformity': cross_uniformity.detach(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        log_dict['train/loss'] = total_loss.detach()

        self._staggered_logger.log(log_dict, self)

        return total_loss

    def get_metrics(
        self,
        audio_features,
        text_features,
        logit_scale_a
    ):
        metrics = {}
    
        logits_per_audio = (logit_scale_a * audio_features @ text_features.t()).detach().cpu()
        logits_per_text = logits_per_audio.t().detach().cpu()

        labels = torch.arange(audio_features.shape[0]).long()
        # Change the loss from two terms into four terms with 2x2 combined CE loss
        total_loss = (
                F.cross_entropy(logits_per_audio, labels)
                + F.cross_entropy(logits_per_text, labels)
        ) / 2

        metrics[f"cumulative_loss"] = total_loss.item()
        metrics[f"num_samples"] = audio_features.shape[0]

        logits = {"audio_to_text": logits_per_audio, "text_to_audio": logits_per_text}

        ground_truth = torch.arange(len(text_features)).view(-1, 1)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == ground_truth)[1]  # (yusong) this line is slow because it uses single thread
            preds = preds.detach().cpu().numpy()
            metrics[f"{name}_mean_rank"] = preds.mean() + 1
            metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
            for k in [1, 5, 10]:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k)
            # map@10
            metrics[f"{name}_mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

        return metrics

    def validation_step(self, batch, batch_idx):
        audios, metadata = batch

        if self.htsat_dataset:
            audio_features = checkpoint(self.model.get_audio_embedding, audios, preprocessed=True)
            prompts = [md["prompt"] for md in metadata]
        else:

            if audios.ndim == 4 and audios.shape[0] == 1:
                audios = audios[0]
                prompts = [md["prompt"][0] for md in metadata]
                padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
            else:
                prompts = [md["prompt"] for md in metadata]
                padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
            
            audios = audios.to(self.device)
            pretransform = self.model.pretransform

            if pretransform is not None:
                pretransform.to(self.device)

                if not self.pre_encoded:
                    with torch.amp.autocast("cuda"):
                        audios = pretransform.encode(audios)

                        padding_masks = resize_padding_mask(padding_masks, audios.shape[2])
                else:
                    if padding_masks.shape[-1] != audios.shape[-1]:
                        padding_masks = resize_padding_mask(padding_masks, audios.shape[-1])

            attention_mask = padding_masks if self.mask_padding_attention else None

            audio_features = self.model.get_audio_embedding(audios, mask=attention_mask)

        text_features = self.model.get_text_embedding(prompts)
        self.all_val_text_features.append(text_features)
        self.all_val_audio_features.append(audio_features)

    def on_validation_epoch_end(self):
        metrics = {}
        all_val_text_features = torch.cat(self.all_val_text_features)
        all_val_audio_features = torch.cat(self.all_val_audio_features)

        all_val_text_features = rearrange(self.all_gather(all_val_text_features), "w b d -> (w b) d")
        all_val_audio_features = rearrange(self.all_gather(all_val_audio_features), "w b d -> (w b) d")

        val_metrics = self.get_metrics(all_val_audio_features, all_val_text_features, torch.exp(self.model.logit_scale_a))

        val_log_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
        self.log_dict(val_log_metrics)

        self.all_val_text_features = []
        self.all_val_audio_features = []


    def export_model(self, path, use_safetensors=False):
                
        if use_safetensors:
            save_file(self.model.state_dict(), path)
        else:
            torch.save({"state_dict": self.model.state_dict()}, path)

class CLAPValidationCallback(pl.Callback):
    def __init__(self, demo_every: int = 1000, demo_dl = None):
        super().__init__()

        self.demo_every = demo_every
        self.last_demo_step = -1

        self.demo_dl = demo_dl

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: CLAPTrainingWrapper, outputs, batch, batch_idx):        

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

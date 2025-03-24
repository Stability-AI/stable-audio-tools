import pytorch_lightning as pl
import sys, gc
import random
import torch
import torchaudio
import typing as tp
import wandb

from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from torch import optim
from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..interface.aeiou import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from ..models.lm import AudioLanguageModelWrapper
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image

class AudioLanguageModelTrainingWrapper(pl.LightningModule):
    def __init__(
            self, 
            model: AudioLanguageModelWrapper, 
            lr = 1e-4, 
            use_ema=False, 
            ema_copy=None,
            optimizer_configs: dict = None,
            pre_encoded=False
        ):
        super().__init__()

        self.model = model

        self.model.pretransform.requires_grad_(False)

        self.model_ema = None
        if use_ema:
            self.model_ema = EMA(self.model, ema_model=ema_copy, beta=0.99, update_every=10)

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "lm": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (0.9, 0.95),
                            "weight_decay": 0.1
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

    def configure_optimizers(self):
        lm_opt_config = self.optimizer_configs['lm']
        opt_lm = create_optimizer_from_config(lm_opt_config['optimizer'], self.model.parameters())

        if "scheduler" in lm_opt_config:
            sched_lm = create_scheduler_from_config(lm_opt_config['scheduler'], opt_lm)
            sched_lm_config = {
                "scheduler": sched_lm,
                "interval": "step"
            }
            return [opt_lm], [sched_lm_config]

        return [opt_lm]
        
    # Copied and modified from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/solvers/musicgen.py under MIT license
    # License can be found in LICENSES/LICENSE_META.txt

    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if not self.pre_encoded:
            codes = self.model.pretransform.tokenize(reals)
        else:
            codes = reals

        padding_masks = []
        for md in metadata:
            if md["padding_mask"].ndim == 1:
                padding_masks.append(md["padding_mask"])
            else:
                padding_masks.append(md["padding_mask"][0])
            
        padding_masks = torch.stack(padding_masks, dim=0).to(self.device) # Shape (batch_size, sequence_length)

        # Interpolate padding masks to the same length as the codes
        padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=codes.shape[2], mode='nearest').bool()
    
        condition_tensors = None

        # If the model is conditioned, get the conditioning tensors
        if self.model.conditioner is not None:
            condition_tensors = self.model.conditioner(metadata, self.device)

        lm_output = self.model.compute_logits(codes, condition_tensors=condition_tensors, cfg_dropout_prob=0.1)

        logits = lm_output.logits # [b, k, t, c]
        logits_mask = lm_output.mask # [b, k, t]

        logits_mask = logits_mask & padding_masks

        cross_entropy, cross_entropy_per_codebook = self._compute_cross_entropy(logits, codes, logits_mask)

        loss = cross_entropy

        log_dict = {
            'train/loss': loss.detach(),
            'train/cross_entropy': cross_entropy.detach(),
            'train/perplexity': torch.exp(cross_entropy).detach(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for k, ce_q in enumerate(cross_entropy_per_codebook):
            log_dict[f'cross_entropy_q{k + 1}'] = ce_q
            log_dict[f'perplexity_q{k + 1}'] = torch.exp(ce_q)

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.model_ema is not None:
            self.model_ema.update()

    def export_model(self, path, use_safetensors=False):
        
        model = self.model_ema.ema_model if self.model_ema is not None else self.model

        if use_safetensors:
            save_file(model.state_dict(), path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)
        

class AudioLanguageModelDemoCallback(pl.Callback):
    def __init__(self, 
                 demo_every=2000,
                 num_demos=8,
                 sample_size=65536,
                 sample_rate=48000,
                 demo_conditioning: tp.Optional[tp.Dict[str, tp.Any]] = None,
                 demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7],
                 **kwargs
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_samples = sample_size
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_conditioning = demo_conditioning
        self.demo_cfg_scales = demo_cfg_scales

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: AudioLanguageModelTrainingWrapper, outputs, batch, batch_idx):        

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        module.eval()

        print(f"Generating demo")
        self.last_demo_step = trainer.global_step

        demo_length_tokens = self.demo_samples // module.model.pretransform.downsampling_ratio

        #demo_reals = batch[0][:self.num_demos]

        # if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
        #     demo_reals = demo_reals[0]

        #demo_reals_tokens = module.model.pretransform.tokenize(demo_reals)

        ##Limit to first 50 tokens
        #demo_reals_tokens = demo_reals_tokens[:, :, :50]

        try:
            print("Getting conditioning")

            for cfg_scale in self.demo_cfg_scales:

                model = module.model # module.model_ema.ema_model if module.model_ema is not None else module.model

                print(f"Generating demo for cfg scale {cfg_scale}")
                fakes = model.generate_audio(
                    batch_size=self.num_demos,
                    max_gen_len=demo_length_tokens, 
                    conditioning=self.demo_conditioning, 
                    #init_data = demo_reals_tokens,
                    cfg_scale=cfg_scale,
                    temp=1.0,
                    top_p=0.95
                )

                # Put the demos together
                fakes = rearrange(fakes, 'b d n -> d (b n)')

                filename = f'demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                fakes = fakes / fakes.abs().max()
                fakes = fakes.type(torch.float32).clamp(-1, 1).mul(32767).type(torch.int16).cpu()
                torchaudio.save(filename, fakes, self.sample_rate)

                log_audio(
                    trainer.logger, f'demo_cfg_{cfg_scale}', filename,
                    sample_rate=self.sample_rate, caption='Reconstructed')
                log_image(
                    trainer.logger, f'demo_melspec_left_cfg_{cfg_scale}',
                    audio_spectrogram_image(fakes))

        except Exception as e:
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()

import pytorch_lightning as pl
import sys, gc
import random
import torch
import torchaudio
import typing as tp
import wandb

from aeiou.viz import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from ema_pytorch import EMA
from einops import rearrange
from torch import optim
from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout, ConditioningAttributes

from time import time

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


class MusicGenTrainingWrapper(pl.LightningModule):
    def __init__(self, musicgen_model, lr = 1e-4, ema_copy=None):
        super().__init__()

        self.musicgen_model: MusicGen = musicgen_model

        self.musicgen_model.compression_model.requires_grad_(False)

        self.lm = self.musicgen_model.lm

        self.lm.to(torch.float32).train().requires_grad_(True)

        self.lm_ema = EMA(self.lm, ema_model=ema_copy, beta=0.99, update_every=10)

        self.cfg_dropout = ClassifierFreeGuidanceDropout(0.1)

        self.lr = lr

    def configure_optimizers(self):
        optimizer = optim.AdamW([*self.lm.parameters()], lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)

        return optimizer
    
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

        # Convert reals to mono if necessary
        if self.musicgen_model.audio_channels == 1:
            reals = reals.mean(dim=1, keepdim=True)

        self.musicgen_model.compression_model.to(self.device).eval()
        self.lm.to(self.device).train()
        self.lm.condition_provider.to(self.device).eval()

        self.lm.condition_provider.conditioners["description"].device = self.device
        self.lm.condition_provider.conditioners["description"].t5.to(self.device).eval()

        with torch.cuda.amp.autocast():

            codes, _ = self.musicgen_model.compression_model.encode(reals) # [b, k, t]

            attributes = [ConditioningAttributes(text={'description': md["prompt"][0][:512]}) for md in metadata]
            attributes = self.lm.cfg_dropout(attributes)
            attributes = self.lm.att_dropout(attributes)
            tokenized = self.lm.condition_provider.tokenize(attributes)
     
            with torch.cuda.amp.autocast(enabled=False):
                condition_tensors = self.lm.condition_provider(tokenized)
                
            lm_output = self.lm.compute_predictions(
                codes=codes,
                conditions = [],
                condition_tensors = condition_tensors,
            )

            logits = lm_output.logits # [b, k, t, c]
            logits_mask = lm_output.mask # [b, k, t]

            cross_entropy, cross_entropy_per_codebook = self._compute_cross_entropy(logits, codes, logits_mask)

            loss = cross_entropy

        log_dict = {
            'train/loss': loss.detach(),
            'train/cross_entropy': cross_entropy.detach(),
            'train/perplexity': torch.exp(cross_entropy).detach(),
        }

        for k, ce_q in enumerate(cross_entropy_per_codebook):
            log_dict[f'cross_entropy_q{k + 1}'] = ce_q
            log_dict[f'perplexity_q{k + 1}'] = torch.exp(ce_q)

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.lm_ema.update()

    def export_model(self, path):
        self.musicgen_model.lm = self.lm_ema.ema_model
        export_state_dict = {"state_dict": self.musicgen_model.state_dict()}
        
        torch.save(export_state_dict, path)

class MusicGenDemoCallback(pl.Callback):
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
    def on_train_batch_end(self, trainer, module: MusicGenTrainingWrapper, outputs, batch, batch_idx):        

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        module.eval()

        print(f"Generating demo")
        self.last_demo_step = trainer.global_step

        demo_length_sec = self.demo_samples // self.sample_rate

        try:
            print("Getting conditioning")

            prompts = [md["prompt"][:512] for md in self.demo_conditioning]

            for cfg_scale in self.demo_cfg_scales:

                module.musicgen_model.set_generation_params(duration=demo_length_sec, cfg_coef=cfg_scale)

                print(f"Generating demo for cfg scale {cfg_scale}")
                fakes = module.musicgen_model.generate(prompts, progress=True)

                # Put the demos together
                fakes = rearrange(fakes, 'b d n -> d (b n)')

                log_dict = {}
                
                filename = f'demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fakes, self.sample_rate)

                log_dict[f'demo_cfg_{cfg_scale}'] = wandb.Audio(filename,
                                                    sample_rate=self.sample_rate,
                                                    caption=f'Reconstructed')
            
                log_dict[f'demo_melspec_left_cfg_{cfg_scale}'] = wandb.Image(audio_spectrogram_image(fakes))

                trainer.logger.experiment.log(log_dict)

        except Exception as e:
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()
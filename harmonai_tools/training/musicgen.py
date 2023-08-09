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
from pytorch_lightning.utilities.distributed import rank_zero_only

from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout

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
    def __init__(self, musicgen_model, lr = 1e-4):
        super().__init__()

        self.musicgen_model: MusicGen = musicgen_model

        self.musicgen_model.compression_model.requires_grad_(False)

        self.lm = self.musicgen_model.lm

        self.lm.to(torch.float32).train().requires_grad_(True)

        self.cfg_dropout = ClassifierFreeGuidanceDropout(0.1)

        self.lr = lr

    def configure_optimizers(self):
        optimizer = optim.AdamW([*self.lm.parameters()], lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1)

        return optimizer

    def training_step(self, batch, batch_idx):
        reals, metadata = batch
        reals = reals[0]

        # Convert reals to mono
        reals = reals.mean(dim=1, keepdim=True)

        condition_strings = [md["prompt"][0] for md in metadata]

        self.musicgen_model.compression_model.to(self.device).eval()
        self.lm.to(self.device).train()
        self.lm.condition_provider.to(self.device).eval()

        self.lm.condition_provider.conditioners["description"].device = self.device
        self.lm.condition_provider.conditioners["description"].t5.to(self.device).eval()

        codes, _ = self.musicgen_model.compression_model.encode(reals) # [b, k, t]
        
        # Get the conditioning attributes from the prompts
        conditioning, _ = self.musicgen_model._prepare_tokens_and_attributes(condition_strings, None)

        # Apply dropout to the conditioning attributes
        conditioning = self.cfg_dropout(conditioning)

        # tokenize the conditioning attributes
        tokenized = self.lm.condition_provider.tokenize(conditioning)

        condition_tensors = self.lm.condition_provider(tokenized)
        
        with torch.cuda.amp.autocast():
            lm_output = self.lm.compute_predictions(
                codes=codes,
                conditions = [],
                condition_tensors = condition_tensors,
            )

            logits = lm_output.logits # [b, k, t, c]
            logits_mask = lm_output.mask # [b, k, t]

            logits = logits.float()
            logits_mask = logits_mask.float()

            #one-hot encode the codes
            codes = F.one_hot(codes, num_classes=self.lm.card).float() # [b, k, t, c]
            
            # Flatten and mask the logits
            logits = logits.reshape(-1, logits.shape[-1])
            logits_mask = logits_mask.reshape(-1)
            logits = logits[logits_mask == 1]
            codes = codes.reshape(-1, codes.shape[-1])
            codes = codes[logits_mask == 1]

            loss = F.cross_entropy(logits, codes.argmax(dim=-1))

        log_dict = {
            'train/loss': loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def export_model(self, path):
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

            prompts = [md["prompt"] for md in self.demo_conditioning]

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


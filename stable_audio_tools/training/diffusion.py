import pytorch_lightning as pl
import sys, gc
import random
import torch
import torchaudio
import typing as tp
import wandb

import auraloss
from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from torch import optim
from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..interface.aeiou import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler, truncated_logistic_normal_rescaled, DistributionShift
from ..models.diffusion import DiffusionModelWrapper, ConditionedDiffusionModelWrapper
from ..models.autoencoders import DiffusionAutoencoder
from ..models.diffusion_prior import PriorType
from .autoencoders import create_loss_modules_from_bottleneck
from .losses import AuralossLoss, MSELoss, MultiLoss
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image, log_metric, log_point_cloud

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

class DiffusionUncondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training an unconditional audio diffusion model (like Dance Diffusion).
    '''
    def __init__(
            self,
            model: DiffusionModelWrapper,
            lr: float = 1e-4,
            pre_encoded: bool = False
    ):
        super().__init__()

        self.diffusion = model

        self.diffusion_ema = EMA(
            self.diffusion.model,
            beta=0.9999,
            power=3/4,
            update_every=1,
            update_after_step=1
        )

        self.lr = lr

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        loss_modules = [
            MSELoss("v",
                     "targets",
                     weight=1.0,
                     name="mse_loss"
                )
        ]

        self.losses = MultiLoss(loss_modules)

        self.pre_encoded = pre_encoded

    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=self.lr)

    def training_step(self, batch, batch_idx):
        reals = batch[0]

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        loss_info = {}

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        if self.diffusion.pretransform is not None:
            if not self.pre_encoded:
                with torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        loss_info["reals"] = diffusion_input

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas
        targets = noise * alphas - diffusion_input * sigmas

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_inputs, t)

            loss_info.update({
                "v": v,
                "targets": targets
            })

            loss, losses = self.losses(loss_info)

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):

        self.diffusion.model = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionUncondDemoCallback(pl.Callback):
    def __init__(self,
                 demo_every=2000,
                 num_demos=8,
                 demo_steps=250,
                 sample_rate=48000
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

        demo_samples = module.diffusion.sample_size

        if module.diffusion.pretransform is not None:
            demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

        noise = torch.randn([self.num_demos, module.diffusion.io_channels, demo_samples]).to(module.device)

        try:
            with torch.cuda.amp.autocast():
                fakes = sample(module.diffusion_ema, noise, self.demo_steps, 0)

                if module.diffusion.pretransform is not None:
                    fakes = module.diffusion.pretransform.decode(fakes)

            # Put the demos together
            fakes = rearrange(fakes, 'b d n -> d (b n)')

            filename = f'demo_{trainer.global_step:08}.wav'
            fakes = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)

            log_audio(
                trainer.logger, "demo", filename,
                sample_rate=self.sample_rate, caption='Reconstructed')
            log_image(
                trainer.logger, "demo_melspec_left",
                audio_spectrogram_image(fakes))

            del fakes
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
        finally:
            gc.collect()
            torch.cuda.empty_cache()

class DiffusionCondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = None,
            mask_padding: bool = False,
            mask_padding_dropout: float = 0.0,
            use_ema: bool = True,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal"] = "uniform",
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9]
    ):
        super().__init__()

        self.diffusion = model

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler     

        self.diffusion_objective = model.diffusion_objective

        self.loss_modules = [
            MSELoss("output",
                   "targets",
                   weight=1.0,
                   mask_key="padding_mask" if self.mask_padding else None,
                   name="mse_loss"
            )
        ]

        self.losses = MultiLoss(self.loss_modules)

        self.log_loss_info = log_loss_info

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
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

        self.pre_encoded = pre_encoded

        # Validation

        self.validation_timesteps = validation_timesteps

        self.validation_step_outputs = {}

        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        p = Profiler()

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        p.tick("setup")

        #with torch.amp.autocast(device_type="cuda"):
        conditioning = self.diffusion.conditioner(metadata, self.device)

        # If mask_padding is on, randomly drop the padding masks to allow for learning silence padding
        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout

        # Create batch tensor of attention masks from the "mask" field of the metadata array
        if use_padding_mask:
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        p.tick("conditioning")

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)

                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    p.tick("pretransform")

                    # If mask_padding is on, interpolate the padding masks to the size of the pretransformed input
                    if use_padding_mask:
                        padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
        elif self.timestep_sampler == "trunc_logit_normal":
            # Draw from logistic truncated normal distribution
            t = truncated_logistic_normal_rescaled(reals.shape[0]).to(self.device)

            # Flip the distribution
            t = 1 - t
        else:
            raise ValueError(f"Invalid timestep_sampler: {self.timestep_sampler}")

        if self.diffusion.dist_shift is not None:
            # Shift the distribution
            t = self.diffusion.dist_shift.time_shift(t, reals.shape[2])

        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective in ["v"]:
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input

        p.tick("noise")

        extra_args = {}

        if use_padding_mask:
            extra_args["mask"] = padding_masks

        output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, **extra_args)
        p.tick("diffusion")

        loss_info.update({
            "output": output,
            "targets": targets,
            "padding_mask": padding_masks if use_padding_mask else None,
        })

        loss, losses = self.losses(loss_info)

        p.tick("loss")

        if self.log_loss_info:
            # Loss debugging logs
            num_loss_buckets = 10
            bucket_size = 1 / num_loss_buckets
            loss_all = F.mse_loss(output, targets, reduction="none")

            sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze()

            # gather loss_all across all GPUs
            loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")

            # Bucket loss values based on corresponding sigma values, bucketing sigma values by bucket_size
            loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean() for i in torch.arange(0, 1, bucket_size).to(self.device)])

            # Log bucketed losses with corresponding sigma bucket values, if it's not NaN
            debug_log_dict = {
                f"model/loss_all_{i/num_loss_buckets:.1f}": loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
            }

            self.log_dict(debug_log_dict)


        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        p.tick("log")
        #print(f"Profiler: {p}")
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def validation_step(self, batch, batch_idx):

        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        with torch.cuda.amp.autocast() and torch.no_grad():
            conditioning = self.diffusion.conditioner(metadata, self.device)

        # TODO: decide what to do with padding masks during validation

        # # If mask_padding is on, randomly drop the padding masks to allow for learning silence padding
        # use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout

        # # Create batch tensor of attention masks from the "mask" field of the metadata array
        # if use_padding_mask:
        #     padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.no_grad():
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)

                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)

                    # # If mask_padding is on, interpolate the padding masks to the size of the pretransformed input
                    # if use_padding_mask:
                    #     padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        for validation_timestep in self.validation_timesteps:

            t = torch.full((reals.shape[0],), validation_timestep, device=self.device)

            # Calculate the noise schedule parameters for those timesteps
            if self.diffusion_objective in ["v"]:
                alphas, sigmas = get_alphas_sigmas(t)
            elif self.diffusion_objective == "rectified_flow":
                alphas, sigmas = 1-t, t

            # Combine the ground truth data and the noise
            alphas = alphas[:, None, None]
            sigmas = sigmas[:, None, None]
            noise = torch.randn_like(diffusion_input)
            noised_inputs = diffusion_input * alphas + noise * sigmas

            if self.diffusion_objective == "v":
                targets = noise * alphas - diffusion_input * sigmas
            elif self.diffusion_objective == "rectified_flow":
                targets = noise - diffusion_input

            extra_args = {}

            # if use_padding_mask:
            #     extra_args["mask"] = padding_masks

            with torch.cuda.amp.autocast() and torch.no_grad():
                output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = 0, **extra_args)

                val_loss = F.mse_loss(output, targets)

                self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'].append(val_loss.item())

    def on_validation_epoch_end(self):
        log_dict = {}
        for validation_timestep in self.validation_timesteps:
            outputs_key = f'val/loss_{validation_timestep:.1f}'
            val_loss = sum(self.validation_step_outputs[outputs_key]) / len(self.validation_step_outputs[outputs_key])

            # Gather losses across all GPUs
            val_loss = self.all_gather(val_loss).mean().item()

            log_metric(self.logger, outputs_key, val_loss, step=self.global_step)

        # Get average over all timesteps
        val_loss = torch.tensor([val for val in self.validation_step_outputs.values()]).mean()

        # Gather losses across all GPUs
        val_loss = self.all_gather(val_loss).mean().item()

        log_metric(self.logger, 'val/avg_loss', val_loss, step=self.global_step)

        # Reset validation losses
        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []


    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionCondDemoCallback(pl.Callback):
    def __init__(self,
                 demo_every=2000,
                 num_demos=8,
                 sample_size=65536,
                 demo_steps=250,
                 sample_rate=48000,
                 demo_conditioning: tp.Optional[tp.Dict[str, tp.Any]] = {},
                 demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7],
                 demo_cond_from_batch: bool = False,
                 display_audio_cond: bool = False,
                 cond_display_configs: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_samples = sample_size
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_conditioning = demo_conditioning
        self.demo_cfg_scales = demo_cfg_scales

        # If true, the callback will use the metadata from the batch to generate the demo conditioning
        self.demo_cond_from_batch = demo_cond_from_batch

        # If true, the callback will display the audio conditioning
        self.display_audio_cond = display_audio_cond

        self.cond_display_configs = cond_display_configs

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionCondTrainingWrapper, outputs, batch, batch_idx):

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        module.eval()

        print(f"Generating demo")
        self.last_demo_step = trainer.global_step

        demo_samples = self.demo_samples

        demo_cond = self.demo_conditioning

        if self.demo_cond_from_batch:
            # Get metadata from the batch
            demo_cond = batch[1][:self.num_demos]

        if module.diffusion.pretransform is not None:
            demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

        noise = torch.randn([self.num_demos, module.diffusion.io_channels, demo_samples]).to(module.device)

        try:
            print("Getting conditioning")
            with torch.cuda.amp.autocast():
                conditioning = module.diffusion.conditioner(demo_cond, module.device)

            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            if self.display_audio_cond:
                audio_inputs = torch.cat([cond["audio"] for cond in demo_cond], dim=0)
                audio_inputs = rearrange(audio_inputs, 'b d n -> d (b n)')

                filename = f'demo_audio_cond_{trainer.global_step:08}.wav'
                audio_inputs = audio_inputs.to(torch.float32).div(torch.max(torch.abs(audio_inputs))).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, audio_inputs, self.sample_rate)
                log_audio(trainer.logger, f'demo_audio_cond', filename, self.sample_rate)
                log_image(trainer.logger, f"demo_audio_cond_melspec_left", audio_spectrogram_image(audio_inputs))

            # Pre-generation conditioning display
            if self.cond_display_configs is not None:
                for cond_display_config in self.cond_display_configs:
                    cond_id = cond_display_config.get("id", None)
                    assert cond_id is not None, "cond_display_configs must have an 'id' field"

                    cond_type = cond_display_config.get("type", None)
                    assert cond_type is not None, "cond_display_configs must have a 'type' field"

                    if cond_type == "audio":
                        audio_cond_config = cond_display_config.get("config", {})
                        is_pre_encoded = audio_cond_config.get("pre_encoded", False)
                        audio_inputs = torch.stack([cond[cond_id] for cond in demo_cond], dim=0)

                        if is_pre_encoded:
                            # Decode the pre-encoded audio conditioning
                            audio_inputs = module.diffusion.pretransform.decode(audio_inputs)

                        audio_inputs_out = rearrange(audio_inputs, 'b d n -> d (b n)')
                        filename = f'demo_{cond_id}_{trainer.global_step:08}.wav'
                        audio_inputs_out = audio_inputs_out.to(torch.float32).div(torch.max(torch.abs(audio_inputs_out))).mul(32767).to(torch.int16).cpu()
                        torchaudio.save(filename, audio_inputs_out, self.sample_rate)
                        log_audio(trainer.logger, f'demo_{cond_id}', filename, self.sample_rate)
                        log_image(trainer.logger, f"demo_{cond_id}_melspec_left", audio_spectrogram_image(audio_inputs_out))

            for cfg_scale in self.demo_cfg_scales:

                print(f"Generating demo for cfg scale {cfg_scale}")

                with torch.cuda.amp.autocast():
                    model = module.diffusion_ema.model if module.diffusion_ema is not None else module.diffusion.model

                    if module.diffusion_objective == "v":
                        fakes = sample(model, noise, self.demo_steps, 0, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)
                    elif module.diffusion_objective == "rectified_flow":
                        fakes = sample_discrete_euler(model, noise, self.demo_steps, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)

                    if module.diffusion.pretransform is not None:
                        fakes = module.diffusion.pretransform.decode(fakes)

                # Put the demos together
                fakes = rearrange(fakes, 'b d n -> d (b n)')

                filename = f'demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                fakes_out = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fakes_out, self.sample_rate)
                log_audio(trainer.logger, f'demo_cfg_{cfg_scale}', filename, self.sample_rate)                
                log_image(trainer.logger, f'demo_melspec_left_cfg_{cfg_scale}', audio_spectrogram_image(fakes_out))
            
                # Mid-generation conditioning display
                if self.cond_display_configs is not None:
                    for cond_display_config in self.cond_display_configs:
                        cond_id = cond_display_config.get("id", None)
                        assert cond_id is not None, "cond_display_configs must have an 'id' field"

                        cond_type = cond_display_config.get("type", None)
                        assert cond_type is not None, "cond_display_configs must have a 'type' field"

                        if cond_type == "audio":
                            audio_cond_config = cond_display_config.get("config", {})
                            display_mix = audio_cond_config.get("display_mix", False)
                            if display_mix:
                                is_pre_encoded = audio_cond_config.get("pre_encoded", False)
                                audio_inputs = torch.stack([cond[cond_id] for cond in demo_cond], dim=0)

                                if is_pre_encoded:
                                    # Decode the pre-encoded audio conditioning
                                    audio_inputs = module.diffusion.pretransform.decode(audio_inputs)

                                filename = f'demo_{cond_id}_mix_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                                audio_inputs = rearrange(audio_inputs, 'b d n -> d (b n)')
                                audio_mix = audio_inputs + fakes
                                audio_mix_out = audio_mix.to(torch.float32).div(torch.max(torch.abs(audio_mix))).mul(32767).to(torch.int16).cpu()
                                torchaudio.save(filename, audio_mix_out, self.sample_rate)
                                log_audio(trainer.logger, f'demo_{cond_id}_mix_cfg_{cfg_scale}', filename, self.sample_rate)

                        elif cond_type == "audio_dict":
                            audio_cond_config = cond_display_config.get("config", {})
                            display_mix = audio_cond_config.get("display_mix", False)
                            is_pre_encoded = audio_cond_config.get("pre_encoded", False)

                            submixes = []

                            for cond in demo_cond:

                                audio_dict = cond[cond_id]
                                audio_inputs = torch.stack([audio_dict[key] for key in audio_dict.keys()], dim=0)

                                if is_pre_encoded:
                                    # Decode the pre-encoded audio conditioning
                                    audio_inputs = module.diffusion.pretransform.decode(audio_inputs)
                            
                                submix = torch.sum(audio_inputs, dim=0)
                                submixes.append(submix)

                            submix = torch.stack(submixes, dim=0)
                            submix = rearrange(submix, 'b d n -> d (b n)')
                            filename = f'demo_{cond_id}_submix_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                            submix_out = submix.to(torch.float32).div(torch.max(torch.abs(submix))).mul(32767).to(torch.int16).cpu()
                            torchaudio.save(filename, submix_out, self.sample_rate)
                            log_audio(trainer.logger, f'demo_{cond_id}_submix_cfg_{cfg_scale}', filename, self.sample_rate)

                            filename = f'demo_{cond_id}_mix_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                            audio_mix = submix + fakes
                            audio_mix_out = audio_mix.to(torch.float32).div(torch.max(torch.abs(audio_mix))).mul(32767).to(torch.int16).cpu()
                            torchaudio.save(filename, audio_mix_out, self.sample_rate)
                            log_audio(trainer.logger, f'demo_{cond_id}_mix_cfg_{cfg_scale}', filename, self.sample_rate)

            del fakes

        except Exception as e:
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()

class DiffusionCondInpaintTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = 1e-4,
            max_mask_segments = 10,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            use_ema: bool = True,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal"] = "uniform",
    ):
        super().__init__()

        self.diffusion = model

        self.use_ema = use_ema

        if self.use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.cfg_dropout_prob = cfg_dropout_prob

        self.lr = lr
        self.max_mask_segments = max_mask_segments

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler

        self.diffusion_objective = model.diffusion_objective

        self.loss_modules = [
            MSELoss("output",
                   "targets",
                   weight=1.0,
                   name="mse_loss"
            )
        ]

        self.losses = MultiLoss(self.loss_modules)

        self.log_loss_info = log_loss_info

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
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

        self.pre_encoded = pre_encoded

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]

    def random_mask(self, sequence, max_mask_length, padding_masks):
        b, _, sequence_length = sequence.size()

        # Create a mask tensor for each batch element
        masks = []

        for i in range(b):
            mask_type = random.randint(0, 2)
            padding_mask = padding_masks[i]

            # Get index of first element in padding mask that is 0
            real_sequence_length = torch.argmax((padding_mask == 0).to(torch.int16)).item()

            if real_sequence_length == 0:
                real_sequence_length = sequence_length

            if mask_type == 0:  # Random mask with multiple segments
                num_segments = random.randint(1, self.max_mask_segments)
                max_segment_length = max(real_sequence_length // num_segments, 1)

                segment_lengths = []
                for i in range(num_segments):
                    segment_lengths.append(random.randint(1, max_segment_length))
                #segment_lengths = random.sample(range(1, max_segment_length + 1), num_segments)

                mask = torch.ones((1, 1, sequence_length))
                for length in segment_lengths:
                    mask_start = random.randint(0, real_sequence_length - length)
                    mask[:, :, mask_start:mask_start + length] = 0

            elif mask_type == 1:  # Full mask
                mask = torch.zeros((1, 1, sequence_length))

            elif mask_type == 2:  # Causal mask
                mask = torch.zeros((1, 1, sequence_length))
                mask_length = random.randint(1, real_sequence_length)
                mask[:, :, :mask_length] = 1

            mask = mask.to(sequence.device)
            masks.append(mask)

        # Concatenate the mask tensors into a single tensor
        mask = torch.cat(masks, dim=0).to(sequence.device)

        # Apply the mask to the sequence tensor for each batch element
        masked_sequence = sequence * mask

        return masked_sequence, mask

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        p = Profiler()

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        # Get padding masks so we don't place inpainting masks over padding
        if self.pre_encoded:
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
        else:
            padding_masks = torch.stack([md["padding_mask"] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        diffusion_input = reals

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        p.tick("setup")

        #with torch.cuda.amp.autocast():
        conditioning = self.diffusion.conditioner(metadata, self.device)

        p.tick("conditioning")

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                p.tick("pretransform")

                padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        # Max mask size is the full sequence length
        max_mask_length = diffusion_input.shape[2]

        # Create a mask of random length for a random slice of the input
        masked_input, mask = self.random_mask(diffusion_input, max_mask_length, padding_masks)

        conditioning['inpaint_mask'] = [mask]
        conditioning['inpaint_masked_input'] = [masked_input]

        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
        elif self.timestep_sampler == "trunc_logit_normal":
            # Draw from logistic truncated normal distribution
            t = truncated_logistic_normal_rescaled(reals.shape[0]).to(self.device)

            # Flip the distribution
            t = 1 - t
        else:
            raise ValueError(f"Invalid timestep_sampler: {self.timestep_sampler}")

        if self.diffusion.dist_shift is not None:
            # Shift the distribution
            t = self.diffusion.dist_shift.time_shift(t, reals.shape[2])

        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective in ["v"]:
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input

        p.tick("noise")

        extra_args = {}

        #with torch.cuda.amp.autocast():
        p.tick("amp")
        output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, **extra_args)
        p.tick("diffusion")

        loss_info.update({
            "output": output,
            "targets": targets,
        })

        loss, losses = self.losses(loss_info)

        if self.log_loss_info:
            # Loss debugging logs
            num_loss_buckets = 10
            bucket_size = 1 / num_loss_buckets
            loss_all = F.mse_loss(output, targets, reduction="none")

            sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze()

            # gather loss_all across all GPUs
            loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")

            # Bucket loss values based on corresponding sigma values, bucketing sigma values by bucket_size
            loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean() for i in torch.arange(0, 1, bucket_size).to(self.device)])

            # Log bucketed losses with corresponding sigma bucket values, if it's not NaN
            debug_log_dict = {
                f"model/loss_all_{i/num_loss_buckets:.1f}": loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
            }

            self.log_dict(debug_log_dict)

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        p.tick("log")
        #print(f"Profiler: {p}")
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionCondInpaintDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        demo_steps=250,
        sample_size=65536,
        sample_rate=48000,
        num_demos=8,
        demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7]
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_steps = demo_steps
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.demo_cfg_scales = demo_cfg_scales
        self.num_demos = num_demos
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionCondTrainingWrapper, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

        try:

            demo_reals, metadata = next(self.demo_dl)

            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]

            # Limit to num_demos
            demo_reals = demo_reals[:self.num_demos]
            metadata = metadata[:self.num_demos]

            demo_reals = demo_reals.to(module.device)

            if not module.pre_encoded:
                # Log the real audio
                log_image(trainer.logger, f'demo_reals_melspec_left', audio_spectrogram_image(rearrange(demo_reals, "b d n -> d (b n)").mul(32767).to(torch.int16).cpu()))

                if module.diffusion.pretransform is not None:
                    module.diffusion.pretransform.to(module.device)
                    demo_reals = module.diffusion.pretransform.encode(demo_reals)
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(module.diffusion.pretransform, "scale") and module.diffusion.pretransform.scale != 1.0:
                    demo_reals = demo_reals / module.diffusion.pretransform.scale

            demo_samples = demo_reals.shape[2]

            # Get conditioning
            conditioning = module.diffusion.conditioner(metadata, module.device)

            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(module.device) # Shape (batch_size, sequence_length)

            masked_input, mask = module.random_mask(demo_reals, demo_reals.shape[2], padding_masks)

            conditioning['inpaint_mask'] = [mask]
            conditioning['inpaint_masked_input'] = [masked_input]

            if module.diffusion.pretransform is not None:
                log_image(trainer.logger, f'demo_masked_input', tokens_spectrogram_image(masked_input.cpu()))
            else:
                log_image(trainer.logger, f'demo_masked_input', audio_spectrogram_image(rearrange(masked_input, "b c t -> c (b t)").mul(32767).to(torch.int16).cpu()))

            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            noise = torch.randn([demo_reals.shape[0], module.diffusion.io_channels, demo_samples]).to(module.device)

            # Cast the noise to the dtype of the model
            model_dtype = next(module.diffusion.parameters()).dtype
            noise = noise.to(model_dtype)

            for cfg_scale in self.demo_cfg_scales:
                model = module.diffusion_ema.model if module.diffusion_ema is not None else module.diffusion.model
                print(f"Generating demo for cfg scale {cfg_scale}")

                with torch.cuda.amp.autocast():
                    if module.diffusion_objective == "v":
                        fakes = sample(model, noise, self.demo_steps, 0, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)
                    elif module.diffusion_objective == "rectified_flow":
                        fakes = sample_discrete_euler(model, noise, self.demo_steps, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)

                if module.diffusion.pretransform is not None:
                    fakes = module.diffusion.pretransform.decode(fakes)

                # Put the demos together
                fakes = rearrange(fakes, 'b d n -> d (b n)')

                filename = f'demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                fakes = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fakes, self.sample_rate)

                log_audio(trainer.logger, f'demo_cfg_{cfg_scale}', filename, self.sample_rate)
                log_image(trainer.logger, f'demo_melspec_left_cfg_{cfg_scale}', audio_spectrogram_image(fakes))

        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e

class DiffusionAutoencoderTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a diffusion autoencoder
    '''
    def __init__(
            self,
            model: DiffusionAutoencoder,
            lr: float = 1e-4,
            ema_copy = None,
            use_reconstruction_loss: bool = False
    ):
        super().__init__()

        self.diffae = model

        self.diffae_ema = EMA(
            self.diffae,
            ema_model=ema_copy,
            beta=0.9999,
            power=3/4,
            update_every=1,
            update_after_step=1,
            include_online_model=False
        )

        self.lr = lr

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        loss_modules = [
            MSELoss("v",
                    "targets",
                    weight=1.0,
                    name="mse_loss"
            )
        ]

        if model.bottleneck is not None:
            # TODO: Use loss config for configurable bottleneck weights and reconstruction losses
            loss_modules += create_loss_modules_from_bottleneck(model.bottleneck, {})

        self.use_reconstruction_loss = use_reconstruction_loss

        if use_reconstruction_loss:
            scales = [2048, 1024, 512, 256, 128, 64, 32]
            hop_sizes = []
            win_lengths = []
            overlap = 0.75
            for s in scales:
                hop_sizes.append(int(s * (1 - overlap)))
                win_lengths.append(s)

            sample_rate = model.sample_rate

            stft_loss_args = {
                "fft_sizes": scales,
                "hop_sizes": hop_sizes,
                "win_lengths": win_lengths,
                "perceptual_weighting": True
            }

            out_channels = model.out_channels

            if model.pretransform is not None:
                out_channels = model.pretransform.io_channels

            if out_channels == 2:
                self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            else:
                self.sdstft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

            loss_modules.append(
                AuralossLoss(self.sdstft, 'audio_reals', 'audio_pred', name='mrstft_loss', weight=0.1), # Reconstruction loss
            )

        self.losses = MultiLoss(loss_modules)

    def configure_optimizers(self):
        return optim.Adam([*self.diffae.parameters()], lr=self.lr)

    def training_step(self, batch, batch_idx):
        reals = batch[0]

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        loss_info["audio_reals"] = reals

        if self.diffae.pretransform is not None:
            with torch.no_grad():
                reals = self.diffae.pretransform.encode(reals)

        loss_info["reals"] = reals

        #Encode reals, skipping the pretransform since it was already applied
        latents, encoder_info = self.diffae.encode(reals, return_info=True, skip_pretransform=True)

        loss_info["latents"] = latents
        loss_info.update(encoder_info)

        if self.diffae.decoder is not None:
            latents = self.diffae.decoder(latents)

        # Upsample latents to match diffusion length
        if latents.shape[2] != reals.shape[2]:
            latents = F.interpolate(latents, size=reals.shape[2], mode='nearest')

        loss_info["latents_upsampled"] = latents

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        with torch.cuda.amp.autocast():
            v = self.diffae.diffusion(noised_reals, t, input_concat_cond=latents)

            loss_info.update({
                "v": v,
                "targets": targets
            })

            if self.use_reconstruction_loss:
                pred = noised_reals * alphas - v * sigmas

                loss_info["pred"] = pred

                if self.diffae.pretransform is not None:
                    pred = self.diffae.pretransform.decode(pred)
                    loss_info["audio_pred"] = pred

            loss, losses = self.losses(loss_info)

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': reals.std(),
            'train/latent_std': latents.std(),
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.diffae_ema.update()

    def export_model(self, path, use_safetensors=False):

        model = self.diffae_ema.ema_model

        if use_safetensors:
            save_file(model.state_dict(), path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)

class DiffusionAutoencoderDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        demo_steps=250,
        sample_size=65536,
        sample_rate=48000
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_steps = demo_steps
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionAutoencoderTrainingWrapper, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

        demo_reals, _ = next(self.demo_dl)

        # Remove extra dimension added by WebDataset
        if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
            demo_reals = demo_reals[0]

        encoder_input = demo_reals

        encoder_input = encoder_input.to(module.device)

        demo_reals = demo_reals.to(module.device)

        with torch.no_grad() and torch.cuda.amp.autocast():
            latents = module.diffae_ema.ema_model.encode(encoder_input).float()
            fakes = module.diffae_ema.ema_model.decode(latents, steps=self.demo_steps)

        #Interleave reals and fakes
        reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')

        # Put the demos together
        reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')

        filename = f'recon_{trainer.global_step:08}.wav'
        reals_fakes = reals_fakes.to(torch.float32).div(torch.max(torch.abs(reals_fakes))).mul(32767).to(torch.int16).cpu()
        torchaudio.save(filename, reals_fakes, self.sample_rate)

        # log_dict[f'recon'] = wandb.Audio(
        #    filename, sample_rate=self.sample_rate, caption=f'Reconstructed')
        # log_dict[f'embeddings_3dpca'] = pca_point_cloud(latents)
        # log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))
        # log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(reals_fakes))

        log_audio(
            trainer.logger, "recon", filename,
            sample_rate=self.sample_rate, caption='Reconstructed')
        log_point_cloud(
            trainer.logger, "embeddings_3dpca", pca_point_cloud(latents))
        log_image(
            trainer.logger, "embeddings_spec",
            tokens_spectrogram_image(latents))
        log_image(
            trainer.logger, "recon_melspec_left",
            audio_spectrogram_image(reals_fakes))

        if module.diffae_ema.ema_model.pretransform is not None:
            with torch.no_grad() and torch.cuda.amp.autocast():
                initial_latents = module.diffae_ema.ema_model.pretransform.encode(encoder_input)
                first_stage_fakes = module.diffae_ema.ema_model.pretransform.decode(initial_latents)
                first_stage_fakes = rearrange(first_stage_fakes, 'b d n -> d (b n)')
                first_stage_fakes = first_stage_fakes.to(torch.float32).mul(32767).to(torch.int16).cpu()
                first_stage_filename = f'first_stage_{trainer.global_step:08}.wav'
                torchaudio.save(first_stage_filename, first_stage_fakes, self.sample_rate)

                log_audio(
                    trainer.logger, "first_stage", first_stage_filename,
                    sample_rate=self.sample_rate, caption='First Stage Reconstructed')
                log_image(
                    trainer.logger, "first_stage_latents",
                    tokens_spectrogram_image(initial_latents))
                log_image(
                    trainer.logger, "first_stage_melspec_left",
                    audio_spectrogram_image(first_stage_fakes))

def create_source_mixture(reals, num_sources=2):
    # Create a fake mixture source by mixing elements from the training batch together with random offsets
    source = torch.zeros_like(reals)
    for i in range(reals.shape[0]):
        sources_added = 0

        js = list(range(reals.shape[0]))
        random.shuffle(js)
        for j in js:
            if i == j or (i != j and sources_added < num_sources):
                # Randomly offset the mixed element between 0 and the length of the source
                seq_len = reals.shape[2]
                offset = random.randint(0, seq_len-1)
                source[i, :, offset:] += reals[j, :, :-offset]
                if i == j:
                    # If this is the real one, shift the reals as well to ensure alignment
                    new_reals = torch.zeros_like(reals[i])
                    new_reals[:, offset:] = reals[i, :, :-offset]
                    reals[i] = new_reals
                sources_added += 1

    return source

class DiffusionPriorTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a diffusion prior for inverse problems
    Prior types:
        mono_stereo: The prior is conditioned on a mono version of the audio to generate a stereo version
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = 1e-4,
            ema_copy = None,
            prior_type: PriorType = PriorType.MonoToStereo,
            use_reconstruction_loss: bool = False,
            log_loss_info: bool = False,
    ):
        super().__init__()

        self.diffusion = model

        self.diffusion_ema = EMA(
            self.diffusion,
            ema_model=ema_copy,
            beta=0.9999,
            power=3/4,
            update_every=1,
            update_after_step=1,
            include_online_model=False
        )

        self.lr = lr

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.log_loss_info = log_loss_info

        loss_modules = [
            MSELoss("v",
                    "targets",
                    weight=1.0,
                    name="mse_loss"
            )
        ]

        self.use_reconstruction_loss = use_reconstruction_loss

        if use_reconstruction_loss:
            scales = [2048, 1024, 512, 256, 128, 64, 32]
            hop_sizes = []
            win_lengths = []
            overlap = 0.75
            for s in scales:
                hop_sizes.append(int(s * (1 - overlap)))
                win_lengths.append(s)

            sample_rate = model.sample_rate

            stft_loss_args = {
                "fft_sizes": scales,
                "hop_sizes": hop_sizes,
                "win_lengths": win_lengths,
                "perceptual_weighting": True
            }

            out_channels = model.io_channels

            self.audio_out_channels = out_channels

            if model.pretransform is not None:
                out_channels = model.pretransform.io_channels

            if self.audio_out_channels == 2:
                self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
                self.lrstft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

                # Add left and right channel reconstruction losses in addition to the sum and difference
                self.loss_modules += [
                    AuralossLoss(self.lrstft, 'audio_reals_left', 'pred_left', name='stft_loss_left', weight=0.05),
                    AuralossLoss(self.lrstft, 'audio_reals_right', 'pred_right', name='stft_loss_right', weight=0.05),
                ]

            else:
                self.sdstft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

            self.loss_modules.append(
                AuralossLoss(self.sdstft, 'audio_reals', 'audio_pred', name='mrstft_loss', weight=0.1), # Reconstruction loss
            )

        self.losses = MultiLoss(loss_modules)

        self.prior_type = prior_type

    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=self.lr)

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        loss_info["audio_reals"] = reals

        if self.prior_type == PriorType.MonoToStereo:
            source = reals.mean(dim=1, keepdim=True).repeat(1, reals.shape[1], 1).to(self.device)
            loss_info["audio_reals_mono"] = source
        else:
            raise ValueError(f"Unknown prior type {self.prior_type}")

        if self.diffusion.pretransform is not None:
            with torch.no_grad():
                reals = self.diffusion.pretransform.encode(reals)

                if self.prior_type in [PriorType.MonoToStereo]:
                    source = self.diffusion.pretransform.encode(source)

        if self.diffusion.conditioner is not None:
            with torch.cuda.amp.autocast():
                conditioning = self.diffusion.conditioner(metadata, self.device)
        else:
            conditioning = {}

        loss_info["reals"] = reals

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        with torch.cuda.amp.autocast():

            conditioning['source'] = [source]

            v = self.diffusion(noised_reals, t, cond=conditioning, cfg_dropout_prob = 0.1)

            loss_info.update({
                "v": v,
                "targets": targets
            })

            if self.use_reconstruction_loss:
                pred = noised_reals * alphas - v * sigmas

                loss_info["pred"] = pred

                if self.diffusion.pretransform is not None:
                    pred = self.diffusion.pretransform.decode(pred)
                    loss_info["audio_pred"] = pred

                if self.audio_out_channels == 2:
                    loss_info["pred_left"] = pred[:, 0:1, :]
                    loss_info["pred_right"] = pred[:, 1:2, :]
                    loss_info["audio_reals_left"] = loss_info["audio_reals"][:, 0:1, :]
                    loss_info["audio_reals_right"] = loss_info["audio_reals"][:, 1:2, :]

            loss, losses = self.losses(loss_info)

            if self.log_loss_info:
                # Loss debugging logs
                num_loss_buckets = 10
                bucket_size = 1 / num_loss_buckets
                loss_all = F.mse_loss(v, targets, reduction="none")

                sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze()

                # gather loss_all across all GPUs
                loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")

                # Bucket loss values based on corresponding sigma values, bucketing sigma values by bucket_size
                loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean() for i in torch.arange(0, 1, bucket_size).to(self.device)])

                # Log bucketed losses with corresponding sigma bucket values, if it's not NaN
                debug_log_dict = {
                    f"model/loss_all_{i/num_loss_buckets:.1f}": loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
                }

                self.log_dict(debug_log_dict)

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': reals.std()
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):

        #model = self.diffusion_ema.ema_model
        model = self.diffusion

        if use_safetensors:
            save_file(model.state_dict(), path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)

class DiffusionPriorDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        demo_steps=250,
        sample_size=65536,
        sample_rate=48000
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_steps = demo_steps
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionAutoencoderTrainingWrapper, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

        demo_reals, metadata = next(self.demo_dl)

        # Remove extra dimension added by WebDataset
        if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
            demo_reals = demo_reals[0]

        demo_reals = demo_reals.to(module.device)

        encoder_input = demo_reals

        if module.diffusion.conditioner is not None:
            with torch.cuda.amp.autocast():
                conditioning_tensors = module.diffusion.conditioner(metadata, module.device)

        else:
            conditioning_tensors = {}

        with torch.no_grad() and torch.cuda.amp.autocast():
            if module.prior_type == PriorType.MonoToStereo and encoder_input.shape[1] > 1:
                source = encoder_input.mean(dim=1, keepdim=True).repeat(1, encoder_input.shape[1], 1).to(module.device)

            if module.diffusion.pretransform is not None:
                encoder_input = module.diffusion.pretransform.encode(encoder_input)
                source_input = module.diffusion.pretransform.encode(source)
            else:
                source_input = source

            conditioning_tensors['source'] = [source_input]

            fakes = sample(module.diffusion_ema.model, torch.randn_like(encoder_input), self.demo_steps, 0, cond=conditioning_tensors)

            if module.diffusion.pretransform is not None:
                fakes = module.diffusion.pretransform.decode(fakes)

        #Interleave reals and fakes
        reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')
        # Put the demos together
        reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')

        filename = f'recon_{trainer.global_step:08}.wav'
        reals_fakes = reals_fakes.to(torch.float32).div(torch.max(torch.abs(reals_fakes))).mul(32767).to(torch.int16).cpu()
        torchaudio.save(filename, reals_fakes, self.sample_rate)

        log_audio(
            trainer.logger, "recon", filename,
            sample_rate=self.sample_rate, caption='Reconstructed')
        log_image(
            trainer.logger, "recon_melspec_left",
            audio_spectrogram_image(reals_fakes))

        #Log the source
        filename = f'source_{trainer.global_step:08}.wav'
        source = rearrange(source, 'b d n -> d (b n)')
        source = source.to(torch.float32).mul(32767).to(torch.int16).cpu()
        torchaudio.save(filename, source, self.sample_rate)

        log_audio(
            trainer.logger, "source", filename,
            sample_rate=self.sample_rate, caption='Source')
        log_image(
            trainer.logger, "source_melspec_left",
            audio_spectrogram_image(source))

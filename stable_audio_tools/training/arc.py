import math
import pytorch_lightning as pl
import random
import torch
import typing as tp
from functools import partial

from .ema import EMA
from inf_cl import cal_inf_loss
from torch import optim, nn
from torch.distributed.fsdp.wrap import wrap
from torch.nn import functional as F

from ..inference.sampling import truncated_logistic_normal_rescaled, sample_diffusion
from ..models.diffusion import ConditionedDiffusionModelWrapper
from ..models.inpainting import random_inpaint_mask
from .fsdp import recursive_wrap, get_non_fsdp_trainable_params, sync_non_fsdp_gradients
from .utils import create_optimizer_from_config, create_scheduler_from_config, create_augmented_padding_mask, compute_masked_loss, compute_normalized_mse, resize_padding_mask, StaggeredLogger
from ..models.lora import add_lora, get_lora_params, get_lora_layers, get_lora_state_dict, LoRAParametrization, save_lora_safetensors

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def build_congruent_sampler(step_counts, device_fn):
    """Build a congruent power-of-2 discrete timestep sampler.

    Samples from a nested grid of discrete t values where coarser step counts
    (1, 2, 4, ...) use strict subsets of the finest grid. For each sample,
    a step count is chosen uniformly, then a random position within that
    count's schedule is selected. This naturally upweights grid positions
    shared by more step counts (e.g. the first step is trained by all counts).

    Args:
        step_counts: List of power-of-2 step counts, e.g. [1, 2, 4, 8, 16].
        device_fn: Callable returning the current device.
    """
    step_counts = sorted(step_counts)
    max_steps = max(step_counts)

    # Uniform grid in [0,1] — dist_shift handles any warping on top
    query_points = torch.linspace(1.0, 0.0, max_steps + 1)[:-1]  # (max_steps,)

    sc_probs = torch.ones(len(step_counts)) / len(step_counts)
    step_counts_tensor = torch.tensor(step_counts)

    def sample(b):
        device = device_fn()
        sc_probs_d = sc_probs.to(device)
        step_counts_d = step_counts_tensor.to(device)
        query_points_d = query_points.to(device)
        sc_idx = torch.multinomial(sc_probs_d.expand(b, -1), 1).squeeze(1)
        sampled_sc = step_counts_d[sc_idx]
        strides = max_steps // sampled_sc
        pos_idx = (torch.rand(b, device=device) * sampled_sc.float()).long()
        grid_idx = pos_idx * strides
        return query_points_d[grid_idx]

    return sample

def euler_step(x_t, v_t, t, s):
    return x_t + (s - t)[:, None, None] * v_t

def sample_euler(model_wrapper, x_t, t, s, cond, steps=4, **teacher_kwargs):
    # Get linear ramp between t and s
    ts = torch.linspace(0, 1, steps=steps+1, device=x_t.device)
    
    for step in range(steps):
        t_step = t + (s - t) * ts[step]
        t_next_step = t + (s - t) * ts[step+1]
        v_t = checkpoint(model_wrapper, x_t, t_step, cond=cond, **teacher_kwargs)
        x_t = euler_step(x_t, v_t, t_step, t_next_step)
    
    return x_t

def sample_euler_model(model, x_t, t, s, steps=4, **teacher_kwargs):
    # Get linear ramp between t and s
    ts = torch.linspace(0, 1, steps=steps+1, device=x_t.device)
    
    for step in range(steps):
        t_step = t + (s - t) * ts[step]
        t_next_step = t + (s - t) * ts[step+1]
        v_t = checkpoint(model, x_t, t_step, **teacher_kwargs)
        x_t = euler_step(x_t, v_t, t_step, t_next_step)
    
    return x_t

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)  

class ARCTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for ARC post-training on a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            arc_config: dict,
            optimizer_configs: dict,
            discriminator: tp.Optional[ConditionedDiffusionModelWrapper] = None,
            teacher_model: tp.Optional[ConditionedDiffusionModelWrapper] = None,
            use_ema: bool = True,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.0,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal"] = "uniform",
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9],
            clip_grad_norm: float = 0.0,
            trim_config = None,
            inpainting_config = None,
            clap_model = None,
            clap_loss_type: tp.Literal["audio_cosine_sim", "audio_contrastive", "audio_text_distance"] = "audio_cosine_sim",
            mask_padding_attention: bool = False,
            silence_extension_scale_seconds: float = 0.0,
            mask_loss_weight: float = 0.0,
            sample_rate: int = 44100,
            sample_size: int = None,
            use_effective_length_for_schedule: bool = False,
            log_every_n_steps: int = 10,
            loss_normalization: tp.Literal["none", "timestep", "sample", "sample_channel"] = "none",
            loss_norm_eps: float = 1e-6,
            lora_state_dict: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        super().__init__()

        self._staggered_logger = StaggeredLogger(every_n_steps=log_every_n_steps)

        self.sample_size = sample_size

        self.automatic_optimization = False

        self.diffusion = model

        self.reuse_discriminator_as_teacher = (teacher_model is not None and teacher_model is discriminator)

        if self.reuse_discriminator_as_teacher:
            # Don't store teacher as a separate nn.Module attribute to avoid
            # duplicate state_dict entries. During warmup, use self.discriminator.
            self.teacher_model = None
        else:
            self.teacher_model = teacher_model

            if self.teacher_model is not None:
                self.teacher_model.eval().requires_grad_(False)

        # Also keep pretransform in eval mode (it has noise_regularize with train/eval-dependent scale)
        if model.pretransform is not None:
            model.pretransform.eval()

        self.clap_model = clap_model
        self.clap_loss_type = clap_loss_type

        if self.clap_model is not None:
            self.clap_model.eval()

            del self.clap_model.pretransform
            if self.clap_loss_type != "audio_text_distance":
                del self.clap_model.text_branch

            self.clap_loss_weight = 1.0

        self.discriminator = discriminator

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9995,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        # Generator LoRA configuration
        self.gen_lora_config = arc_config.get('lora', None)

        if self.gen_lora_config is not None:
            # Disable EMA with LoRA (not compatible)
            if self.diffusion_ema is not None:
                print("Warning: Disabling EMA since generator LoRA is enabled")
                self.diffusion_ema = None

            # Freeze model and conditioner
            self.diffusion.model.eval().requires_grad_(False)
            self.diffusion.conditioner.eval().requires_grad_(False)

            rank = self.gen_lora_config.get('rank', 8)
            lora_alpha = self.gen_lora_config.get('alpha', rank)
            lora_dropout = self.gen_lora_config.get('dropout', 0.0)
            lora_adapter_type = self.gen_lora_config.get('adapter_type', "lora")

            include = self.gen_lora_config.get('include', None)
            exclude = self.gen_lora_config.get('exclude', None)

            lora_config = {
                torch.nn.Linear: {
                    "weight": partial(
                        LoRAParametrization.from_linear,
                        rank=rank,
                        lora_alpha=lora_alpha,
                        lora_dropout_p=lora_dropout,
                        adapter_type=lora_adapter_type
                    ),
                },
                torch.nn.Conv1d: {
                    "weight": partial(
                        LoRAParametrization.from_conv1d,
                        rank=rank,
                        lora_alpha=lora_alpha,
                        lora_dropout_p=lora_dropout,
                        adapter_type=lora_adapter_type
                    ),
                },
            }

            add_lora(self.diffusion.model, lora_config, include=include, exclude=exclude)
            add_lora(self.diffusion.conditioner, lora_config, include=include, exclude=exclude)

            if lora_state_dict is not None:
                self.diffusion.model.load_state_dict(lora_state_dict, strict=False)
                self.diffusion.conditioner.load_state_dict(lora_state_dict, strict=False)

            # Log info
            num_lora_model = sum(p.numel() for p in get_lora_params(self.diffusion.model))
            num_lora_cond = sum(p.numel() for p in get_lora_params(self.diffusion.conditioner))
            print(f"Generator LoRA enabled: rank={rank}, alpha={lora_alpha}, dropout={lora_dropout}")
            print(f"LoRA layers: model={len(get_lora_layers(self.diffusion.model))}, conditioner={len(get_lora_layers(self.diffusion.conditioner))}")
            print(f"LoRA params: model={num_lora_model:,}, conditioner={num_lora_cond:,}, total={num_lora_model + num_lora_cond:,}")

        self.ode_warmup_config = arc_config.get('ode_warmup', None)

        if self.ode_warmup_config is not None:

            self.ode_warmup_steps = self.ode_warmup_config.get('warmup_steps', 0)
            self.ode_refresh_rate = self.ode_warmup_config.get('refresh_rate', 1)
            self.ode_n_sampling_steps = self.ode_warmup_config.get('sampling_steps', 20)
            self.ode_warmup_cfg = self.ode_warmup_config.get('cfg', 4.0)
        else:
            self.ode_warmup_steps = 0

        self.diff_states = []

        self.noise_dist_config = arc_config.get('noise_dist', {})
        self.gen_noise_dist = self.build_noise_dist('generator')

        if self.clap_model is not None:
            self.clap_noise_dist = self.build_noise_dist('clap')

        self.disc_lora_config = None  # Will be set if discriminator exists and lora config provided

        if self.discriminator is not None:
            self.dis_noise_dist = self.build_noise_dist('discriminator')

            self.discriminator_config = arc_config.get('discriminator', {})

            self.discriminator_dit_layer = self.discriminator_config.get('dit_hidden_layer', None)

            assert self.discriminator_dit_layer is not None, "Must specify discriminator dit_hidden_layer in ARC config"
            
            if isinstance(self.discriminator_dit_layer, int):
                self.discriminator_dit_layer = [self.discriminator_dit_layer]

            self.do_contrastive_disc = self.discriminator_config.get('contrastive', False)

            self.contrastive_beta = self.discriminator_config.get('contrastive_beta', 1.0)
            
            self.disc_hinge_loss = self.discriminator_config.get('disc_hinge_loss', False)

            self.include_grad_penalties = self.discriminator_config.get('include_grad_penalties', False)

            # Option to freeze the discriminator DiT backbone
            self.freeze_discriminator_backbone = self.discriminator_config.get('freeze_backbone', False)

            if self.freeze_discriminator_backbone:
                self.discriminator.eval()
                for param in self.discriminator.parameters():
                    param.requires_grad = False

            self.grad_penalty_weight = self.discriminator_config.get('grad_penalty_weight', 100.0)

            self.include_r2_penalty = self.discriminator_config.get('include_r2_penalty', True)

            discriminator_type = self.discriminator_config.get('type', 'convnext')
            discriminator_model_config = self.discriminator_config.get('config', {})

            # Layer combination mode: 'concat' (default) or 'stack' (for transformer head)
            self.layer_combine_mode = self.discriminator_config.get('layer_combine_mode', 'concat')
            
            dit_dim = self.discriminator.model.model.transformer.dim
            num_layers = len(self.discriminator_dit_layer)
            
            # For 'stack' mode, input dim is just dit_dim (layers become sequence)
            # For 'concat' mode, input dim is dit_dim * num_layers
            if self.layer_combine_mode == 'stack':
                disc_head_input_dim = dit_dim
            else:
                disc_head_input_dim = dit_dim * num_layers

            if discriminator_type == 'convnext':
                from ..models.arc import ConvNeXtDiscriminator
                self.discriminator_head = ConvNeXtDiscriminator(in_channels = disc_head_input_dim, latent_dim=1, **discriminator_model_config)
            elif discriminator_type == 'conv':
                from ..models.arc import ConvDiscriminator
                self.discriminator_head = ConvDiscriminator(channels = disc_head_input_dim, **discriminator_model_config)
            elif discriminator_type == 'dilated_conv':
                from ..models.arc import DilatedConvDiscriminator
                self.discriminator_head = DilatedConvDiscriminator(channels = disc_head_input_dim, **discriminator_model_config)
            elif discriminator_type == 'transformer':
                from ..models.arc import TransformerDiscriminator
                self.discriminator_head = TransformerDiscriminator(channels = disc_head_input_dim, **discriminator_model_config)
            elif discriminator_type == 'multi_scale':
                from ..models.arc import MultiScaleTransformerDiscriminator
                self.discriminator_head = MultiScaleTransformerDiscriminator(channels = disc_head_input_dim, **discriminator_model_config)

            self.multi_scale_disc = (discriminator_type == 'multi_scale')
            if self.multi_scale_disc:
                self.disc_head_names = self.discriminator_head.head_names

            # LoRA for discriminator backbone
            self.disc_lora_config = self.discriminator_config.get('lora', None)

            if self.disc_lora_config is not None:
                rank = self.disc_lora_config.get('rank', 8)
                lora_alpha = self.disc_lora_config.get('alpha', rank)
                lora_dropout = self.disc_lora_config.get('dropout', 0.0)
                lora_adapter_type = self.disc_lora_config.get('adapter_type', "lora")
                disc_include = self.disc_lora_config.get('include', None)
                disc_exclude = self.disc_lora_config.get('exclude', None)

                lora_config = {
                    torch.nn.Linear: {
                        "weight": partial(
                            LoRAParametrization.from_linear,
                            rank=rank,
                            lora_alpha=lora_alpha,
                            lora_dropout_p=lora_dropout,
                            adapter_type=lora_adapter_type
                        ),
                    },
                    torch.nn.Conv1d: {
                        "weight": partial(
                            LoRAParametrization.from_conv1d,
                            rank=rank,
                            lora_alpha=lora_alpha,
                            lora_dropout_p=lora_dropout,
                            adapter_type=lora_adapter_type
                        ),
                    },
                }
                add_lora(self.discriminator.model, lora_config, include=disc_include, exclude=disc_exclude)

                # Freeze only the base weights of LoRA layers (not norms, embeddings, etc.)
                for module in self.discriminator.model.modules():
                    if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
                        module.parametrizations.weight.original.requires_grad_(False)

                # Log LoRA info
                num_lora_params = sum(p.numel() for p in get_lora_params(self.discriminator.model))
                num_other_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad) - num_lora_params
                print(f"Discriminator LoRA enabled: rank={rank}, alpha={lora_alpha}, dropout={lora_dropout}")
                print(f"LoRA layers: {len(get_lora_layers(self.discriminator.model))}")
                print(f"LoRA params: {num_lora_params:,}, other trainable: {num_other_params:,}")

            self.gen_gan_weight = self.discriminator_config.get('weights', {}).get('generator', 1.0)
            self.dis_gan_weight = self.discriminator_config.get('weights', {}).get('discriminator', 1.0)

            if self.do_contrastive_disc:
                self.contrastive_loss_weight = self.discriminator_config.get('weights', {}).get('contrastive', 1.0)

            # Feature matching loss configuration (uses same layers as discriminator)
            self.feature_matching_weight = self.discriminator_config.get('feature_matching_weight', 0.0)

            # Logit centering penalty - prevents unbounded logit drift in relativistic loss
            self.logit_center_weight = self.discriminator_config.get('logit_center_weight', 0.0)

            # Discriminator head reset - periodically reinitialize to prevent overfitting
            self.discriminator_reset_every = self.discriminator_config.get('reset_every', None)

        self.denoiser_loss_weight = arc_config.get('denoiser_loss_weight', 0.0)
        self.denoiser_loss_t_exponent = arc_config.get('denoiser_loss_t_exponent', 0.0)  # Divide denoiser loss by t^k to mitigate t² bias

        self.forward_sample_prob = arc_config.get('forward_sample_prob', 0.0)

        self.disc_update_interval = arc_config.get('disc_update_interval', 2)

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler     

        self.diffusion_objective = model.diffusion_objective

        assert optimizer_configs is not None, "Must specify optimizer_configs in training config"

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

        self.model_last_layer = self.diffusion.model.model.transformer.project_out.weight

        self.clip_grad_norm = clip_grad_norm

        self.trim_config = trim_config

        if self.trim_config is not None:
            self.trim_prob = self.trim_config.get("trim_prob", 0.0)
            self.trim_type = self.trim_config.get("type", "random_item")

        self.inpainting_config = inpainting_config

        if self.inpainting_config is not None:
            self.inpaint_mask_kwargs = self.inpainting_config.get("mask_kwargs", {})
            self.ode_inpaint_mask = None
            self.ode_inpaint_masked_input = None

        # Padding mask settings
        # Backward compat: if passed from training config, propagate to model
        if mask_padding_attention and not self.diffusion.mask_padding_attention:
            import warnings
            warnings.warn("mask_padding_attention in training config is deprecated. Move to model.diffusion config.", FutureWarning)
            self.diffusion.mask_padding_attention = mask_padding_attention
        self.mask_padding_attention = self.diffusion.mask_padding_attention
        self.silence_extension_scale_seconds = silence_extension_scale_seconds
        self.mask_loss_weight = mask_loss_weight
        self.loss_normalization = loss_normalization
        self.loss_norm_eps = loss_norm_eps
        self.sample_rate = sample_rate
        # Calculate downsampling_ratio from pretransform if available
        if model.pretransform is not None:
            self.downsampling_ratio = model.pretransform.downsampling_ratio
        else:
            self.downsampling_ratio = 1

        # Per-element schedule shift based on effective (unpadded) sequence length
        # Backward compat: if passed from training config, propagate to model
        if use_effective_length_for_schedule and not self.diffusion.use_effective_length_for_schedule:
            import warnings
            warnings.warn("use_effective_length_for_schedule in training config is deprecated. Move to model.diffusion config.", DeprecationWarning)
            self.diffusion.use_effective_length_for_schedule = use_effective_length_for_schedule
        self.use_effective_length_for_schedule = self.diffusion.use_effective_length_for_schedule

        # Validation

        self.validation_timesteps = validation_timesteps

        self.validation_step_outputs = {}

        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []

        # Cache generator params for optimizer and grad clipping
        # Plain list won't be saved in checkpoint (not an nn.Parameter or buffer)
        if self.gen_lora_config is not None:
            self._gen_params = [*get_lora_params(self.diffusion.model), *get_lora_params(self.diffusion.conditioner)]
        else:
            self._gen_params = [*self.diffusion.model.parameters(), *self.diffusion.conditioner.parameters()]

    @property
    def _teacher(self):
        """Return the model used as ODE teacher during warmup.

        When reuse_discriminator_as_teacher is set, the discriminator serves
        as the teacher during warmup (under torch.no_grad).
        """
        if self.reuse_discriminator_as_teacher:
            return self.discriminator
        return self.teacher_model

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']

        # For MuonAdamW, pass (name, param) tuples so fused layer patterns can match
        gen_params = self._gen_params
        if diffusion_opt_config['optimizer'].get('type') == 'MuonAdamW' and self.gen_lora_config is None:
            gen_params = [(n, p) for n, p in self.diffusion.named_parameters() if p.requires_grad]

        if self.discriminator is not None:
            disc_opt_config = self.optimizer_configs['discriminator']

            opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], gen_params)

            # Get discriminator parameters - collect as named tuples, strip if not MuonAdamW
            disc_params = list(self.discriminator_head.named_parameters())

            if not self.freeze_discriminator_backbone:
                if self.disc_lora_config is not None:
                    # With LoRA: include all trainable params (LoRA + norms + embeddings)
                    disc_params += [(n, p) for n, p in self.discriminator.named_parameters() if p.requires_grad]
                else:
                    # Without LoRA: include all discriminator params
                    disc_params += list(self.discriminator.named_parameters())

            if disc_opt_config['optimizer'].get('type') != 'MuonAdamW':
                disc_params = [p for _, p in disc_params]

            opt_disc = create_optimizer_from_config(disc_opt_config['optimizer'], disc_params)

            # Pass FSDP module references to MuonAdamW for summon_full_params
            if getattr(self, 'use_fsdp', False):
                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
                from .optims import MuonAdamW
                fsdp_modules = [m for m in self.modules() if isinstance(m, FullyShardedDataParallel)]
                if isinstance(opt_diff, MuonAdamW):
                    opt_diff.fsdp_modules = fsdp_modules
                if isinstance(opt_disc, MuonAdamW):
                    opt_disc.fsdp_modules = fsdp_modules

            if "scheduler" in diffusion_opt_config and "scheduler" in disc_opt_config:
                sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
                sched_diff_config = {
                    "scheduler": sched_diff,
                    "interval": "step"
                }
                sched_disc = create_scheduler_from_config(disc_opt_config['scheduler'], opt_disc)
                sched_disc_config = {
                    "scheduler": sched_disc,
                    "interval": "step"
                }
                return [opt_diff, opt_disc], [sched_diff_config, sched_disc_config]

            return [opt_diff, opt_disc]
        else:
            opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], gen_params)

            # Pass FSDP module references to MuonAdamW for summon_full_params
            if getattr(self, 'use_fsdp', False):
                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
                from .optims import MuonAdamW
                if isinstance(opt_diff, MuonAdamW):
                    fsdp_modules = [m for m in self.modules() if isinstance(m, FullyShardedDataParallel)]
                    opt_diff.fsdp_modules = fsdp_modules

            if "scheduler" in diffusion_opt_config:
                sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
                sched_diff_config = {
                    "scheduler": sched_diff,
                    "interval": "step"
                }
                return [opt_diff], [sched_diff_config]

            return opt_diff

    def sample_t_with_shift(self, noise_dist_fn, batch_size, seq_length, effective_seq_len: tp.Union[int, torch.Tensor, None] = None):
        """
        Sample timesteps using the given noise distribution and apply distribution shift.

        Args:
            noise_dist_fn: Callable that samples timesteps (from build_noise_dist)
            batch_size: Number of timesteps to sample
            seq_length: Total sequence length (used if effective_seq_len is None)
            effective_seq_len: Optional per-element effective lengths (tensor) or scalar

        Returns:
            Tensor of timesteps with shape (batch_size,)
        """
        t = noise_dist_fn(batch_size)

        if self.diffusion.dist_shift is not None:
            # Use effective_seq_len if provided, otherwise fall back to seq_length
            shift_seq_len = effective_seq_len if effective_seq_len is not None else seq_length
            # Shift the distribution
            t = self.diffusion.dist_shift.shift(t, shift_seq_len)

        return t

    def export_lora_safetensors(self, path):
        """Export generator LoRA weights as a safetensors file with embedded config."""
        if self.gen_lora_config is None:
            raise ValueError("No LoRA config -- this wrapper is not in LoRA mode")
        state_dict = {
            **get_lora_state_dict(self.diffusion.model),
            **get_lora_state_dict(self.diffusion.conditioner)
        }
        save_lora_safetensors(state_dict, self.gen_lora_config, path)

    def on_save_checkpoint(self, checkpoint):
        if self.gen_lora_config is not None:
            checkpoint.clear()
            checkpoint['state_dict'] = {
                **get_lora_state_dict(self.diffusion.model),
                **get_lora_state_dict(self.diffusion.conditioner)
            }
            checkpoint['lora_config'] = self.gen_lora_config

    def calculate_clap_loss(self, reals, fakes, t, metadata, padding_mask=None):

        clap_mask_kwargs = {"padding_mask": padding_mask} if padding_mask is not None else {}

        if self.clap_loss_type in ["audio_cosine_sim", "audio_contrastive"]:
            feature_batch = torch.cat([reals, fakes], dim=0)

            batch_t = torch.cat([t, t], dim=0)
            batch_mask_kwargs = {"padding_mask": torch.cat([padding_mask, padding_mask], dim=0)} if padding_mask is not None else {}

            all_clap_embeddings = self.clap_model.get_audio_embedding(feature_batch, noise_levels=batch_t, **batch_mask_kwargs)
            real_embeds, fake_embeds = all_clap_embeddings.chunk(2, dim=0)

            if self.clap_loss_type == "audio_cosine_sim":

                clap_cosine_sims = F.cosine_similarity(real_embeds, fake_embeds, dim=-1)

                clap_loss = 1 - clap_cosine_sims.mean()

            elif self.clap_loss_type == "audio_contrastive":
                clap_loss = cal_inf_loss(real_embeds, fake_embeds)

        elif self.clap_loss_type == "audio_text_distance":
            # For audio-text cosine similarity, we need to get text embeddings
            # Check clap_prompt first, then prompt_text (raw text saved before pre-tokenization), then prompt
            prompts = [md.get('clap_prompt', md.get('prompt_text', md.get('prompt'))) for md in metadata]
            text_embeddings = self.clap_model.get_text_embedding(prompts)
            audio_embeddings = self.clap_model.get_audio_embedding(fakes, noise_levels=t, **clap_mask_kwargs)

            clap_loss = spherical_dist_loss(text_embeddings, audio_embeddings).mean()

        return clap_loss

    def ode_warmup_step(self, diffusion_input, metadata, padding_masks, effective_seq_len=None):

        if self._teacher is not None:
            # ---- Teacher-based warmup path ----
            teacher = self._teacher

            # Force refresh if diff_states hasn't been populated yet (e.g. checkpoint resume)
            refresh_diff_states = self.global_step % self.ode_refresh_rate == 0 or not isinstance(self.diff_states, dict)

            if refresh_diff_states:
                start_noise = torch.randn_like(diffusion_input)

                with torch.no_grad():
                    teacher_conditioning = teacher.conditioner(metadata, self.device)

                if self.inpainting_config is not None:
                    # Create a mask of random length for a random slice of the input
                    inpaint_masked_input, inpaint_mask = random_inpaint_mask(diffusion_input, padding_masks=padding_masks, mask_padding=self.mask_padding_attention, **self.inpaint_mask_kwargs)

                    teacher_conditioning['inpaint_mask'] = [inpaint_mask]
                    teacher_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

                    self.ode_inpaint_mask = inpaint_mask
                    self.ode_inpaint_masked_input = inpaint_masked_input

                self.ode_metadata = metadata
                self.ode_padding_masks = padding_masks
                self.ode_effective_seq_len = effective_seq_len  # Store per-element effective_seq_len

                # Collect intermediate states via callback
                inters_x = []
                inters_t = []

                def collect_intermediates(info):
                    inters_x.append(info['x'].clone())
                    t_val = info.get('t', info.get('sigma'))
                    if not isinstance(t_val, torch.Tensor):
                        t_val = torch.tensor(t_val)
                    inters_t.append(t_val)

                # Pre-process conditioning into backbone-compatible format
                with torch.no_grad():
                    cond_inputs = teacher.get_conditioning_inputs(teacher_conditioning)

                # sample_diffusion is already @torch.no_grad()
                target = sample_diffusion(
                    model=teacher.model,
                    noise=start_noise,
                    cond_inputs=cond_inputs,
                    diffusion_objective=self.diffusion_objective,
                    steps=self.ode_n_sampling_steps,
                    cfg_scale=self.ode_warmup_cfg,
                    conditioning=metadata,
                    sample_rate=self.sample_rate,
                    pretransform=self.diffusion.pretransform,
                    mask_padding_attention=self.mask_padding_attention,
                    use_effective_length_for_schedule=self.use_effective_length_for_schedule,
                    padding_mask=padding_masks if self.mask_padding_attention else None,
                    dist_shift=teacher.sampling_dist_shift,
                    sampler_type=self.ode_warmup_config.get('sampler', 'dpmpp'),
                    batch_cfg=True,
                    callback=collect_intermediates,
                    disable_tqdm=True,
                    decode=False,
                )

                # Stack intermediates: (steps, B, C, T) and (steps,) or (steps, B)
                self.diff_states = {
                    'target': target.detach(),
                    'x': torch.stack(inters_x).detach(),
                    't': torch.stack(inters_t).detach(),
                }

            # Use stored padding masks (not current batch) since x_t comes from stored diff_states
            # which may have a different sequence length than the current batch
            model_padding_mask = self.ode_padding_masks if self.mask_padding_attention else None

            conditioning = self.diffusion.conditioner(self.ode_metadata, self.device)

            if self.inpainting_config is not None:
                conditioning['inpaint_mask'] = [self.ode_inpaint_mask]
                conditioning['inpaint_masked_input'] = [self.ode_inpaint_masked_input]

            batch_size = diffusion_input.shape[0]
            n_stored_steps = self.diff_states['x'].shape[0]

            # Sample random step indices for each batch element
            ixs = torch.randint(0, n_stored_steps, (batch_size,))

            # Get x_t at the sampled steps for each batch element
            x_t = self.diff_states['x'][ixs, torch.arange(batch_size)].to(self.device)

            # Get t values - handle both scalar (shared schedule) and per-element schedules
            stored_t = self.diff_states['t']
            if stored_t.dim() == 1:
                # Shared schedule: stored_t shape (steps,)
                t = stored_t[ixs].to(self.device)
            else:
                # Per-element schedule: stored_t shape (steps, batch)
                t = stored_t[ixs, torch.arange(batch_size)].to(self.device)

            v_t_student = self.diffusion(x_t, t, cond=conditioning, cfg_dropout_prob=self.cfg_dropout_prob, padding_mask=model_padding_mask)
            denoised_student = euler_step(x_t, v_t_student, t, torch.zeros_like(t))

            # Build loss mask excluding inpainting context (matches RF/denoiser loss behavior)
            ode_loss_mask = self.ode_padding_masks.to(torch.bool)
            if self.inpainting_config is not None:
                ode_loss_mask = ode_loss_mask & ~self.ode_inpaint_mask.squeeze(1).to(torch.bool)

            ode_loss_full = compute_normalized_mse(denoised_student, self.diff_states['target'], ode_loss_mask, self.loss_normalization, self.loss_norm_eps)
            # Reweight by 1/t^k to mitigate t² bias (k=0: no correction, k=1: linear t weighting, k=2: pure velocity)
            if self.denoiser_loss_t_exponent > 0:
                t_weight = t.clamp(min=1e-4) ** (-self.denoiser_loss_t_exponent)
                ode_loss_full = ode_loss_full * t_weight[:, None, None]
            ode_mse_loss, _, _ = compute_masked_loss(
                ode_loss_full, ode_loss_mask, self.mask_padding_attention, self.mask_loss_weight
            )

            # Add context reconstruction loss when varlen training with inpainting
            if self.inpainting_config is not None and self.mask_padding_attention and self.mask_loss_weight > 0:
                inpaint_context = self.ode_inpaint_mask.squeeze(1).to(torch.bool) & self.ode_padding_masks.to(torch.bool)
                n_ctx = inpaint_context.sum(dim=1) * ode_loss_full.shape[1]
                if n_ctx.sum() > 0:
                    context_vals = torch.where(inpaint_context.unsqueeze(1), ode_loss_full, 0.0)
                    ode_mse_loss = ode_mse_loss + (context_vals.sum(dim=(1, 2)) / (n_ctx + 1e-8)).mean() * self.mask_loss_weight

        else:
            # ---- No-teacher warmup path: plain RF denoiser loss ----
            model_padding_mask = padding_masks if self.mask_padding_attention else None
            conditioning = self.diffusion.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                inpaint_masked_input, inpaint_mask = random_inpaint_mask(
                    diffusion_input, padding_masks=padding_masks,
                    mask_padding=self.mask_padding_attention, **self.inpaint_mask_kwargs)
                conditioning['inpaint_mask'] = [inpaint_mask]
                conditioning['inpaint_masked_input'] = [inpaint_masked_input]

            ode_loss_mask = padding_masks.to(torch.bool)
            if self.inpainting_config is not None:
                ode_loss_mask = ode_loss_mask & ~inpaint_mask.squeeze(1).to(torch.bool)

            t = self.sample_t_with_shift(self.gen_noise_dist, diffusion_input.shape[0],
                                          diffusion_input.shape[2], effective_seq_len)
            noise = torch.randn_like(diffusion_input)
            x_t = diffusion_input * (1-t)[:, None, None] + noise * t[:, None, None]

            v_t_student = self.diffusion(x_t, t, cond=conditioning,
                                         cfg_dropout_prob=self.cfg_dropout_prob,
                                         padding_mask=model_padding_mask)
            denoised_student = euler_step(x_t, v_t_student, t, torch.zeros_like(t))

            rf_loss_full = compute_normalized_mse(denoised_student, diffusion_input, ode_loss_mask,
                                                  self.loss_normalization, self.loss_norm_eps)
            if self.denoiser_loss_t_exponent > 0:
                t_weight = t.clamp(min=1e-4) ** (-self.denoiser_loss_t_exponent)
                rf_loss_full = rf_loss_full * t_weight[:, None, None]
            ode_mse_loss, _, _ = compute_masked_loss(
                rf_loss_full, ode_loss_mask, self.mask_padding_attention, self.mask_loss_weight)

        return ode_mse_loss

    def _filter_hidden_states(self, hidden_states):
        """Keep only hidden states needed for discriminator scoring.
        Strips prepended memory tokens so shapes match padding_mask."""
        num_memory = getattr(self.discriminator.model.model.transformer, 'num_memory_tokens', 0)
        filtered = [hidden_states[ix] for ix in self.discriminator_dit_layer]
        if num_memory > 0:
            filtered = [h[:, num_memory:, :] for h in filtered]
        return filtered

    def _disc_scores(self, x, t, cond, padding_mask=None, use_checkpoint=False, return_hidden=False):
        """Run discriminator forward and return scores with filtered hidden states."""
        if use_checkpoint:
            _, info = checkpoint(self.discriminator, x, t, cond=cond, padding_mask=padding_mask, return_info=True)
        else:
            _, info = self.discriminator(x, t, cond=cond, padding_mask=padding_mask, return_info=True)
        hidden = self._filter_hidden_states(info["hidden_states"])
        del info
        scores = self.get_disc_scores(hidden)
        if return_hidden:
            return scores, hidden
        return scores

    def get_disc_scores(self, filtered_hidden_states):
        if self.layer_combine_mode == 'stack':
            # Stack layers as sequence: (B, T, dim) per layer -> (B, num_layers * T, dim)
            stacked = torch.stack(filtered_hidden_states, dim=1)
            B, num_layers, T, dim = stacked.shape
            disc_hidden_states = stacked.reshape(B, num_layers * T, dim)
        else:
            # Concat mode: (B, T, dim * num_layers)
            disc_hidden_states = torch.cat(filtered_hidden_states, dim=-1)
        result = checkpoint(self.discriminator_head, disc_hidden_states.transpose(1, 2))
        # Normalize to list for uniform downstream handling
        if isinstance(result, list):
            return result
        return [result]

    def _compute_score_masks(self, padding_mask, score_tensors):
        """Derive score masks from padding_mask and actual score tensor shapes.

        Uses adaptive_max_pool1d for downsampled outputs so that a score position
        is valid if ANY input position in its receptive field is valid.  Works for
        all discriminator types (conv, dilated conv, transformer, multi-scale).

        Args:
            padding_mask: (B, T_input) bool mask for the input sequence.
            score_tensors: list of score tensors, each (B, 1, T_out).

        Returns a list of (B, T_out) boolean masks, or None if padding_mask is None.
        """
        if padding_mask is None:
            return None

        masks = []
        T_in = padding_mask.shape[-1]
        for score in score_tensors:
            T_out = score.shape[-1]
            if T_out == T_in:
                masks.append(padding_mask)
            else:
                # Downsample: position valid if any input in its window is valid
                mask_float = padding_mask.float().unsqueeze(1)  # (B, 1, T_in)
                mask_down = F.adaptive_max_pool1d(mask_float, T_out).squeeze(1).bool()
                masks.append(mask_down)
        return masks

    @staticmethod
    def _masked_mean(values, mask):
        """Compute mean over valid (masked) positions. Returns .mean() if mask is None."""
        if mask is None:
            return values.mean()
        return (values * mask).sum() / mask.sum().clamp(min=1)

    def calculate_disc_loss(self, real_scores_list, fake_scores_list, score_masks=None, beta=1.0):

        # Relativistic discriminator loss across all scales
        total_loss = 0.0
        for i, (real_scores, fake_scores) in enumerate(zip(real_scores_list, fake_scores_list)):
            diff = (real_scores - fake_scores) * beta

            if self.disc_hinge_loss:
                loss_map = F.relu(1.0 - diff)
            else:
                loss_map = F.softplus(-diff)

            if score_masks is not None:
                mask = score_masks[i].unsqueeze(1)  # (B, 1, T_out)
                total_loss += (loss_map * mask).sum() / mask.sum().clamp(min=1)
            else:
                total_loss += loss_map.mean()

        total_loss = (total_loss / len(real_scores_list)) * self.dis_gan_weight

        return total_loss

    def calculate_feature_matching_loss(self, real_hidden_states, fake_hidden_states, padding_mask=None):
        """
        Compute feature matching loss between real and fake hidden states.
        Matches features at the discriminator layers for GAN stability.
        Uses L1 loss with mean reduction over signal regions only (respects padding mask).

        Args:
            real_hidden_states: Pre-filtered list of tensors (only selected layers), shape (B, T, C)
            fake_hidden_states: Pre-filtered list of tensors (only selected layers), shape (B, T, C)
            padding_mask: Optional boolean tensor (B, T) where True = signal, False = padding

        Returns:
            Scalar feature matching loss
        """
        fm_loss = 0.0

        for real_feat, fake_feat in zip(real_hidden_states, fake_hidden_states):
            # real_feat and fake_feat have shape (B, T, C)

            if padding_mask is not None:
                # Zero out padding regions in hidden states (B, T, 1)
                mask_expanded = padding_mask.unsqueeze(-1)  # (B, T, 1)
                real_feat = real_feat * mask_expanded
                fake_feat = fake_feat * mask_expanded

                # Compute L1 loss only over signal regions
                l1_diff = torch.abs(fake_feat - real_feat.detach())  # (B, T, C)

                # Sum over signal regions and normalize by number of signal elements
                signal_elements = mask_expanded.sum()  # Total number of True positions across batch
                if signal_elements.item() > 0:
                    fm_loss = fm_loss + l1_diff.sum() / (signal_elements * real_feat.shape[-1])
                else:
                    # All padding - no loss contribution
                    pass
            else:
                # No masking - use standard L1 loss
                fm_loss = fm_loss + F.l1_loss(fake_feat, real_feat.detach())

        return fm_loss / len(real_hidden_states)

    def wrap_fsdp(self):
        """
        Wrap model components for FSDP distributed training.

        Uses selective wrapping: only transformer layers are wrapped with FSDP for
        sharding efficiency. Small modules (embeddings, convs, conditioners) are left
        unwrapped to avoid checkpoint synchronization issues.

        Non-FSDP trainable parameters are tracked and their gradients are synchronized
        manually via dist.all_reduce() in manual_backward_with_sync().

        We wrap submodules individually instead of the entire training wrapper to avoid
        FSDP + manual optimization state machine issues.
        See: PyTorch Lightning issues #19626, #19685, #20138
        """
        from ..models.transformer import TransformerBlock

        def wrap_policy(module, name):
            """Wrap only transformer layers (largest modules)."""
            return isinstance(module, TransformerBlock)

        # Wrap generator transformer layers
        recursive_wrap(self.diffusion.model.model, wrap_policy)

        # Wrap discriminator transformer layers if present
        if self.discriminator is not None:
            recursive_wrap(self.discriminator.model.model, wrap_policy)

    def manual_backward_with_sync(self, loss):
        """
        Perform backward pass and sync gradients for non-FSDP trainable parameters.

        FSDP only synchronizes gradients for wrapped modules. Non-wrapped trainable
        parameters need explicit gradient synchronization via dist.all_reduce().
        Call this instead of self.manual_backward() when using selective FSDP wrapping.
        """
        self.manual_backward(loss)

        # Sync gradients for non-FSDP trainable params
        # Compute lazily to avoid storing Parameter references (causes checkpoint issues)
        if getattr(self, 'use_fsdp', False):
            non_fsdp_params = get_non_fsdp_trainable_params(self)
            if non_fsdp_params:
                sync_non_fsdp_gradients(non_fsdp_params)

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast(), torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)

                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    padding_masks = resize_padding_mask(padding_masks, diffusion_input.shape[2])
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        # Apply augmented padding mask (extend valid region to include random silence)
        # This happens after resize to latent size
        if self.mask_padding_attention and self.silence_extension_scale_seconds > 0:
            padding_masks = create_augmented_padding_mask(
                padding_masks,
                silence_extension_scale_seconds=self.silence_extension_scale_seconds,
                sample_rate=self.sample_rate,
                downsampling_ratio=self.downsampling_ratio,
            )

        # Compute effective sequence lengths if enabled (for distribution shift)
        if self.use_effective_length_for_schedule:
            # Use per-element effective lengths derived from seconds_total
            # This matches inference which computes effective length from seconds_total conditioning
            # Fall back to padding_masks.sum() if seconds_total is not available
            if all("seconds_total" in md for md in metadata):
                effective_seq_len = torch.tensor(
                    [int(math.ceil(int(md["seconds_total"] * self.sample_rate) / self.downsampling_ratio)) for md in metadata],
                    device=self.device
                )
            else:
                # Fallback: use padding mask sum
                effective_seq_len = padding_masks.sum(dim=-1)
        else:
            effective_seq_len = None

        if self.discriminator is not None:
            opt_gen, opt_disc = self.optimizers()
        else:
            opt_gen = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        if lr_schedulers is not None and self.discriminator is not None:
            sched_gen, sched_disc = lr_schedulers
        elif lr_schedulers is not None:
            sched_gen = lr_schedulers
        else:
            sched_gen = sched_disc = None

        log_dict = {}

        # Transition from ODE warmup to ARC: free cached diff_states
        if self.ode_warmup_steps > 0 and self.global_step == self.ode_warmup_steps and isinstance(self.diff_states, dict):
            self.diff_states = []
            if self.reuse_discriminator_as_teacher and self.trainer.is_global_zero:
                print(f"Step {self.global_step}: ODE warmup complete, discriminator was reused as teacher")

        if self.global_step < self.ode_warmup_steps:
            warmup_loss = self.ode_warmup_step(diffusion_input, metadata, padding_masks, effective_seq_len)

            opt_gen.zero_grad()
            self.manual_backward_with_sync(warmup_loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self._gen_params, self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                sched_gen.step()

            log_dict = {'train/ode_mse_loss': warmup_loss.detach()}
            self._staggered_logger.log(log_dict, self)

            if self.diffusion_ema is not None:
                if getattr(self, 'use_fsdp', False):
                    from .fsdp import update_ema_fsdp
                    update_ema_fsdp(self.diffusion_ema, self.diffusion.model)
                else:
                    self.diffusion_ema.update()

            return warmup_loss

        # Start trimming after ODE warmup to avoid sequence length issues
        if self.trim_config is not None:
            if random.random() < self.trim_prob:
                
                data_lengths = (torch.sum(padding_masks, dim=1) - 1).tolist()
                
                if self.trim_type == "random_item":
                    trim_length = max(random.choice(data_lengths), 128)

                diffusion_input = diffusion_input[:, :, :trim_length]
                padding_masks = padding_masks[:, :trim_length]

        conditioning = self.diffusion.conditioner(metadata, self.device)

        if self.inpainting_config is not None:

            # Create a mask of random length for a random slice of the input
            inpaint_masked_input, inpaint_mask = random_inpaint_mask(diffusion_input, padding_masks=padding_masks, mask_padding=self.mask_padding_attention, **self.inpaint_mask_kwargs)

            conditioning['inpaint_mask'] = [inpaint_mask]
            conditioning['inpaint_masked_input'] = [inpaint_masked_input]

        # Build loss mask: signal positions, excluding inpainting context
        # (matches diffusion.py behavior where loss is only on the generation region)
        loss_mask = padding_masks.to(torch.bool)
        if self.inpainting_config is not None:
            loss_mask = loss_mask & ~inpaint_mask.squeeze(1).to(torch.bool)

        t = self.sample_t_with_shift(self.gen_noise_dist, reals.shape[0], diffusion_input.shape[2], effective_seq_len)

        gen_noise = torch.randn_like(diffusion_input)

        # Determine padding_mask to pass to model (None if not using mask_padding_attention)
        model_padding_mask = padding_masks if self.mask_padding_attention else None

        # Forward sample: train generator on accumulated-error inputs (like inference)
        # diffusion_input stays as real ground truth for RF loss and discriminator real comparison
        gen_input = diffusion_input
        if self.forward_sample_prob > 0 and random.random() < self.forward_sample_prob:
            with torch.no_grad():
                t_ones = torch.ones(reals.shape[0], device=self.device)
                v_forward = self.diffusion(gen_noise, t_ones, cond=conditioning, cfg_dropout_prob=self.cfg_dropout_prob, padding_mask=model_padding_mask)
                gen_input = euler_step(gen_noise, v_forward, t_ones, torch.zeros_like(t_ones))
                gen_noise = torch.randn_like(gen_input)

        x_t = gen_input * (1-t)[:, None, None] + gen_noise * t[:, None, None]

        train_gen = ((self.global_step % self.disc_update_interval) != (self.disc_update_interval - 1)) or self.discriminator is None

        if train_gen or self.global_step < self.ode_warmup_steps:
            v_t_student = checkpoint(self.diffusion, x_t, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, padding_mask=model_padding_mask)
        else:
            x_t = x_t.requires_grad_(False)
            with torch.no_grad():
                v_t_student = checkpoint(self.diffusion, x_t, t, cond=conditioning, padding_mask=model_padding_mask).detach()

        denoised_student = euler_step(x_t, v_t_student, t, torch.zeros_like(t))

        # Periodically reset discriminator head to prevent overfitting
        if self.discriminator is not None and self.discriminator_reset_every is not None and (self.global_step % self.discriminator_reset_every) == 0 and self.global_step > 0:
            self.discriminator_head.reset_parameters()

        if train_gen:
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

            if self.diffusion_ema is not None:
                if getattr(self, 'use_fsdp', False):
                    from .fsdp import update_ema_fsdp
                    update_ema_fsdp(self.diffusion_ema, self.diffusion.model)
                else:
                    self.diffusion_ema.update()

            if self.discriminator is not None:
                # Get discriminator scores for adversarial loss
                t_gan = self.sample_t_with_shift(self.dis_noise_dist, reals.shape[0], diffusion_input.shape[2], effective_seq_len)
                
                noise = torch.randn_like(denoised_student)
                x_t_gan = denoised_student * (1-t_gan)[:, None, None] + noise * t_gan[:, None, None]

                fake_conditioning = self.discriminator.conditioner(metadata, self.device)

                if self.inpainting_config is not None:
                    fake_conditioning['inpaint_mask'] = [inpaint_mask]
                    fake_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

                # Zero out padding region in both real and fake to ensure discriminator
                # judges based on signal region only (no asymmetry to exploit)
                if self.mask_padding_attention:
                    padding_zero_mask = padding_masks.unsqueeze(1)  # (B, 1, T)
                    x_t_gan = x_t_gan * padding_zero_mask

                # Get fake and real scores (with hidden states for feature matching)
                # Inputs are already zeroed in padding regions (lines 928, 938)
                disc_scores_list, fake_hidden_states = self._disc_scores(x_t_gan, t_gan, fake_conditioning, padding_mask=model_padding_mask, return_hidden=True)

                # Compute per-head score masks from actual score tensor shapes
                score_masks = self._compute_score_masks(model_padding_mask, disc_scores_list)

                x_t_gan_real = diffusion_input * (1-t_gan)[:, None, None] + noise * t_gan[:, None, None]
                if self.mask_padding_attention:
                    x_t_gan_real = x_t_gan_real * padding_zero_mask

                disc_scores_real_list, real_hidden_states = self._disc_scores(x_t_gan_real, t_gan, fake_conditioning, padding_mask=model_padding_mask, return_hidden=True)

                # Relativistic adversarial loss across all scales (masked to signal region)
                loss_adv = 0.0
                for i, (s_real, s_fake) in enumerate(zip(disc_scores_real_list, disc_scores_list)):
                    loss_map = F.softplus(s_real - s_fake)
                    if score_masks is not None:
                        mask = score_masks[i].unsqueeze(1)
                        loss_adv = loss_adv + (loss_map * mask).sum() / mask.sum().clamp(min=1)
                    else:
                        loss_adv = loss_adv + loss_map.mean()
                loss_adv = (loss_adv / len(disc_scores_list)) * self.gen_gan_weight
                log_dict['train/adv_loss'] = loss_adv.detach()

                # Feature matching loss
                if self.feature_matching_weight > 0:
                    # Pass padding_masks if using mask_padding_attention
                    fm_padding_mask = padding_masks if self.mask_padding_attention else None
                    fm_loss = self.calculate_feature_matching_loss(real_hidden_states, fake_hidden_states, padding_mask=fm_padding_mask) * self.feature_matching_weight
                    log_dict['train/feature_matching_loss'] = fm_loss.detach()
                else:
                    fm_loss = torch.tensor(0.0, device=self.device)
            else:
                loss_adv = torch.tensor(0.0, device=self.device)
                fm_loss = torch.tensor(0.0, device=self.device)

            if self.clap_model is not None:
                t_clap = self.sample_t_with_shift(self.clap_noise_dist, reals.shape[0], diffusion_input.shape[2], effective_seq_len)
                noise = torch.randn_like(denoised_student)
                x_t_clap = denoised_student * (1-t_clap)[:, None, None] + noise * t_clap[:, None, None]
                real_t_clap = diffusion_input * (1-t_clap)[:, None, None] + noise * t_clap[:, None, None]
                clap_padding_mask = padding_masks if self.mask_padding_attention else None
                clap_loss = self.calculate_clap_loss(real_t_clap, x_t_clap, t_clap, metadata, padding_mask=clap_padding_mask) * self.clap_loss_weight
                log_dict['train/clap_loss'] = clap_loss.detach()
            else:
                clap_loss = 0

            loss = loss_adv + clap_loss + fm_loss

            opt_gen.zero_grad()

            self.manual_backward_with_sync(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self._gen_params, self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                sched_gen.step()

            log_dict['train/gen_loss'] = loss.detach()
        else:
            denoised_student = denoised_student.detach().requires_grad_(True)
            t_gan = self.sample_t_with_shift(self.dis_noise_dist, reals.shape[0], diffusion_input.shape[2], effective_seq_len)
            noise = torch.randn_like(denoised_student)
            reals_t_gan = diffusion_input * (1-t_gan)[..., None, None] + noise * t_gan[..., None, None]
            denoised_t_gan = denoised_student * (1-t_gan)[..., None, None] + noise * t_gan[..., None, None]

            # Zero out padding region in both real and fake to ensure discriminator
            # judges based on signal region only (no asymmetry to exploit)
            if self.mask_padding_attention:
                padding_zero_mask = padding_masks.unsqueeze(1)  # (B, 1, T)
                reals_t_gan = reals_t_gan * padding_zero_mask
                denoised_t_gan = denoised_t_gan * padding_zero_mask

            reals_t_gan = reals_t_gan.detach().requires_grad_(True)
            denoised_t_gan = denoised_t_gan.detach().requires_grad_(True)

            fake_conditioning = self.discriminator.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                fake_conditioning['inpaint_mask'] = [inpaint_mask]
                fake_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

            disc_reals_output, disc_reals_info = checkpoint(self.discriminator, reals_t_gan, t_gan, cond=fake_conditioning, padding_mask=model_padding_mask, return_info=True)
            disc_reals_hidden = self._filter_hidden_states(disc_reals_info["hidden_states"])
            del disc_reals_info
            disc_scores_reals = self.get_disc_scores(disc_reals_hidden)

            disc_scores_denoised = self._disc_scores(denoised_t_gan, t_gan, fake_conditioning, padding_mask=model_padding_mask, use_checkpoint=True)

            # Compute per-head score masks from actual score tensor shapes
            score_masks = self._compute_score_masks(model_padding_mask, disc_scores_reals)

            if self.include_grad_penalties:

                r1_approx_variance = 0.01

                noised_reals_t_gan = reals_t_gan + r1_approx_variance * torch.randn_like(reals_t_gan)
                disc_scores_noised_reals = self._disc_scores(noised_reals_t_gan, t_gan, fake_conditioning, padding_mask=model_padding_mask, use_checkpoint=True)

                r1_penalty = 0.0
                for i, (s_noised, s_real) in enumerate(zip(disc_scores_noised_reals, disc_scores_reals)):
                    sq_diff = (s_noised - s_real) ** 2
                    if score_masks is not None:
                        mask = score_masks[i].unsqueeze(1)  # (B, 1, T_out)
                        # Per-sample mean over valid positions only
                        r1_penalty = r1_penalty + (sq_diff * mask).sum(dim=[1, 2]) / mask.sum(dim=[1, 2]).clamp(min=1)
                    else:
                        r1_penalty = r1_penalty + torch.mean(sq_diff, dim=[1, 2])
                r1_penalty = (r1_penalty / len(disc_scores_reals)) * self.grad_penalty_weight
                log_dict['r1_penalty'] = r1_penalty.mean().detach()

                if self.include_r2_penalty:
                    noised_denoised_t_gan = denoised_t_gan + r1_approx_variance * torch.randn_like(denoised_t_gan)
                    disc_scores_noised_denoised = self._disc_scores(noised_denoised_t_gan, t_gan, fake_conditioning, padding_mask=model_padding_mask, use_checkpoint=True)
                    r2_penalty = 0.0
                    for i, (s_noised, s_fake) in enumerate(zip(disc_scores_noised_denoised, disc_scores_denoised)):
                        sq_diff = (s_noised - s_fake) ** 2
                        if score_masks is not None:
                            mask = score_masks[i].unsqueeze(1)
                            r2_penalty = r2_penalty + (sq_diff * mask).sum(dim=[1, 2]) / mask.sum(dim=[1, 2]).clamp(min=1)
                        else:
                            r2_penalty = r2_penalty + torch.mean(sq_diff, dim=[1, 2])
                    r2_penalty = (r2_penalty / len(disc_scores_denoised)) * self.grad_penalty_weight
                    log_dict['r2_penalty'] = r2_penalty.mean().detach()
                    grad_penalty_loss = (r1_penalty.mean() + r2_penalty.mean()) / 2
                else:
                    grad_penalty_loss = r1_penalty.mean()

                log_dict['train/grad_penalty_loss'] = grad_penalty_loss.detach()
            else:
                grad_penalty_loss = torch.tensor(0.0, device=self.device)

            # Log disc score diagnostics (gap = separation quality, center = drift from origin)
            if score_masks is not None:
                all_real_valid = torch.cat([s[m.unsqueeze(1).expand_as(s)] for s, m in zip(disc_scores_reals, score_masks)])
                all_fake_valid = torch.cat([s[m.unsqueeze(1).expand_as(s)] for s, m in zip(disc_scores_denoised, score_masks)])
            else:
                all_real_valid = torch.cat([s.reshape(-1) for s in disc_scores_reals])
                all_fake_valid = torch.cat([s.reshape(-1) for s in disc_scores_denoised])
            real_mean = all_real_valid.mean().detach()
            fake_mean = all_fake_valid.mean().detach()
            log_dict['disc_score_gap'] = real_mean - fake_mean
            log_dict['disc_score_center'] = (real_mean + fake_mean) / 2
            if getattr(self, 'multi_scale_disc', False):
                for name, s_r, s_f in zip(self.disc_head_names, disc_scores_reals, disc_scores_denoised):
                    log_dict[f'disc_gap_{name}'] = (s_r.mean() - s_f.mean()).detach()

            loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_denoised, score_masks=score_masks)

            # Logit centering penalty - prevents both logits from drifting unboundedly
            # In relativistic loss, only (real - fake) matters, so both can float to ±∞
            if self.logit_center_weight > 0:
                logit_center_penalty = 0.0
                for i, (s_r, s_f) in enumerate(zip(disc_scores_reals, disc_scores_denoised)):
                    lc_map = (s_r + s_f).pow(2)
                    if score_masks is not None:
                        mask = score_masks[i].unsqueeze(1)
                        logit_center_penalty = logit_center_penalty + (lc_map * mask).sum() / mask.sum().clamp(min=1)
                    else:
                        logit_center_penalty = logit_center_penalty + lc_map.mean()
                logit_center_penalty = (logit_center_penalty / len(disc_scores_reals)) * self.logit_center_weight
                log_dict['train/logit_center_penalty'] = logit_center_penalty.detach()
            else:
                logit_center_penalty = 0

            if self.do_contrastive_disc:

                rolled_metadata = []

                for i in range(reals.shape[0]):
                    rolled_keys = ["prompt"]
                    rolled_metadata.append(metadata[i])
                    for rolled_key in rolled_keys:
                        rolled_metadata[i][rolled_key] = metadata[(i + 1) % reals.shape[0]][rolled_key]

                rolled_conditioning = self.discriminator.conditioner(rolled_metadata, self.device)

                if self.inpainting_config is not None:
                    # Hold inpainting conditioning constant during contrastive conditioning
                    rolled_conditioning['inpaint_mask'] = [inpaint_mask]
                    rolled_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

                disc_scores_rolled_reals = self._disc_scores(reals_t_gan, t_gan, rolled_conditioning, padding_mask=model_padding_mask, use_checkpoint=True)

                contrastive_loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_rolled_reals, score_masks=score_masks, beta=self.contrastive_beta) * self.contrastive_loss_weight

                log_dict['train/contrastive_loss_dis'] = contrastive_loss_dis.detach()
            else:
                contrastive_loss_dis = 0

            loss = loss_dis + contrastive_loss_dis + grad_penalty_loss + logit_center_penalty

            log_dict['train/dis_loss'] = loss_dis.detach()

            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']
            log_dict['train/discriminator_loss'] = loss.detach()
            opt_disc.zero_grad()

            self.manual_backward_with_sync(loss)
            if self.clip_grad_norm > 0.0:
                # Only clip discriminator backbone gradients if not frozen
                disc_params = list(self.discriminator_head.parameters())

                if not self.freeze_discriminator_backbone:
                    if self.disc_lora_config is not None:
                        # With LoRA: include all trainable params
                        disc_params += [p for p in self.discriminator.parameters() if p.requires_grad]
                    else:
                        # Without LoRA: include all discriminator params
                        disc_params += list(self.discriminator.parameters())

                torch.nn.utils.clip_grad_norm_(disc_params, self.clip_grad_norm)
            opt_disc.step()

            if sched_disc is not None:
                sched_disc.step()

        log_dict['train/std_data'] = diffusion_input.std()

        self._staggered_logger.log(log_dict, self)

        return loss

    def build_noise_dist(self, key):
        dist = self.noise_dist_config.get(key, 'uniform')
        if dist == 'uniform':
            return lambda b: self.rng.draw(b)[:, 0].to(self.device)
        elif dist == 'logit_normal':
            return lambda b: torch.sigmoid(torch.randn(b, device=self.device))
        elif dist == 'trunc_logit_normal':
            return lambda b: 1 - truncated_logistic_normal_rescaled(b).to(self.device)
        elif dist == 'one_shot':
            return lambda b: torch.ones(b, device=self.device)
        elif dist == 'denoised':
            return lambda b: torch.zeros(b, device=self.device)
        elif dist == 'logsnr_uniform':
            min_logsnr = -6
            max_logsnr = 2
            return lambda b: torch.sigmoid(-(torch.rand(b, device=self.device) * (max_logsnr - min_logsnr) + min_logsnr))
        elif dist == 'congruent':
            step_counts = self.noise_dist_config.get('congruent_step_counts', [1, 2, 4, 8, 16])
            return build_congruent_sampler(step_counts, lambda: self.device)
        else:
            raise ValueError(f"Invalid noise distribution: {dist}")

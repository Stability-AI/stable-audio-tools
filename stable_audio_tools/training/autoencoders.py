import os
import random
import torch
import torchaudio
import wandb
import math
import pytorch_lightning as pl
from typing import Optional, Literal
import numpy as np

from ..models.autoencoders import AudioAutoencoder, fold_channels_into_batch, unfold_channels_from_batch
from ..models.discriminators import EncodecDiscriminator, OobleckDiscriminator, DACGANLoss, BigVGANDiscriminator, HILDiscriminator, LatentAudioCritic, LatentJEPA, MultiTransformerDiscriminator
from ..models.bottleneck import VAEBottleneck, RVQBottleneck, DACRVQBottleneck, DACRVQVAEBottleneck, RVQVAEBottleneck, WassersteinBottleneck, SoftNormBottleneck
from ..models.dit import DiffusionTransformer
from ..models.transformer import RunningInstanceNorm
from ..models.transforms import TightSpectrogram, ILDTransform, MeanChannelLog1pTransform
from ..models.conditioners import MultiConditioner, T5GemmaConditioner
from ..inference.sampling import truncated_logistic_normal_rescaled
from ..models.pretransforms import WaveletPretransform, PatchedPretransform

from .losses import MelSpectrogramLoss, MultiLoss, AuralossLoss, ValueLoss, L1Loss, LossWithTarget, MSELoss, HubertLoss, CLAPLoss, TimeDomainMMDLoss
from .losses import auraloss as auraloss
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image, log_metric, log_point_cloud, logger_project_name, StaggeredLogger

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ..interface.aeiou import audio_spectrogram_image, tokens_spectrogram_image
from .ema import EMA
from einops import rearrange
from contextlib import nullcontext

def set_requires_grad(model, req_grad: bool):
    for p in model.parameters():
        p.requires_grad = req_grad

def trim_to_shortest(a, b):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:,:,:b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:,:,:a.shape[-1]]
    return a, b

def zero_pad_to_longest(a, b):
    """Zero pad the shorter of two tensors to the length of the longer one."""
    if a.shape[-1] > b.shape[-1]:
        pad_amount = a.shape[-1] - b.shape[-1]
        b = torch.nn.functional.pad(b, (0, pad_amount))
    elif b.shape[-1] > a.shape[-1]:
        pad_amount = b.shape[-1] - a.shape[-1]
        a = torch.nn.functional.pad(a, (0, pad_amount))
    return a, b

def create_batch_from_json(json, key):
    batched = []
    for b in json:
        batched.append(b[key].squeeze(0))
    return torch.stack(batched, dim = 0)

@torch.no_grad()
def calc_update_to_weight_ratio(module, lr: float, scaler=None):
    """Compute update-to-weight ratio without concatenating all params into a single tensor.

    Accumulates squared norms on GPU to avoid the ~2 GB temporary allocation from
    torch.cat that caused CUDA memory fragmentation over thousands of steps.
    Only calls .item() once at the end (single CUDA sync).
    """
    grad_sq = torch.tensor(0.0, device='cuda')
    param_sq = torch.tensor(0.0, device='cuda')
    for p in module.parameters():
        if p.requires_grad:
            # .norm() returns a scalar tensor on the same device — no allocation
            param_sq += p.float().norm().square()
            if p.grad is not None:
                grad_sq += p.grad.float().norm().square()
    # Single CUDA sync to bring both values to CPU
    grad_norm = grad_sq.sqrt()
    if scaler is not None:
        grad_norm = grad_norm / scaler.get_scale()
    upd_norm = grad_norm * lr
    theta_norm = param_sq.sqrt().clamp_min(1e-12)
    return (upd_norm / theta_norm).item()


import math
import torch
from typing import Optional

@torch.no_grad()
def append_random_linear_chirps(
    batch: torch.Tensor,
    sr: float,
    n_chirps: int = 2,
    *,
    fmin: float = 100.0,
    fmax: Optional[float] = 22000.0,
    min_octaves: float = 2.0,
    max_octaves: float = 6.5,
    amp_db_range: tuple[float, float] = (-24.0, -6.0),
    same_chirp_across_channels: bool = True,
    add_dither: float = 1e-4,
    seed: Optional[int] = None,
):
    """
    Concatenate `n_chirps` random linear chirps to `batch` along dim=0.

    Accepts mono [B, N] or multichannel [B, C, N] input and returns the same rank.
    Parameters are sampled so that f0,f1 ∈ [fmin,fmax] (default fmax = 0.45*sr).

    Returns
    -------
    batch_aug : Tensor
        Concatenation of the original batch and the synthetic chirps along batch dim.
        Shape is [B + n_chirps, N] or [B + n_chirps, C, N] to match input.
    meta : dict
        {
          'idx_range': (start_idx, end_idx),  # slice of appended chirps in batch_aug
          'x_chirps': Tensor,                 # the generated chirps alone
          'theta': Tensor,                    # instantaneous phase per sample
          'f0': Tensor, 'f1': Tensor, 'k': Tensor, 'A': Tensor, 'phi': Tensor,
          'valid_sweep': BoolTensor           # True if actual sweep (not tone)
        }
        For multichannel:
          - if same_chirp_across_channels=True: f0,f1,k,A,phi are [n_chirps]
          - else: f0,f1,k,A,phi are [n_chirps, C] (per-channel parameters)
    """
    if n_chirps <= 0:
        return batch

    assert batch.ndim in (2, 3), "batch must be [B,N] or [B,C,N]"
    device, dtype = batch.device, batch.dtype

    if batch.ndim == 2:
        B, N = batch.shape
        C = None
    else:
        B, C, N = batch.shape

    if fmax is None:
        fmax = 0.45 * sr
    assert 0.0 < fmin < fmax < 0.5 * sr, "Require 0 < fmin < fmax < Nyquist."

    # RNG helpers (optional reproducibility)
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)
        def rand(shape): return torch.rand(shape, generator=g, device=device, dtype=dtype)
        def rand_scalar(): return torch.rand((), generator=g, device=device, dtype=dtype)
        def randn_like(t): return torch.randn_like(t, generator=g)
    else:
        def rand(shape): return torch.rand(shape, device=device, dtype=dtype)
        def rand_scalar(): return torch.rand((), device=device, dtype=dtype)
        def randn_like(t): return torch.randn_like(t)

    # Time axis (use T=(N-1)/sr for exact end-point spacing; use N/sr in k to avoid div by 0)
    n = torch.arange(N, device=device, dtype=dtype)
    t = n / sr
    T = (N - 1) / sr if N > 1 else 1.0 / sr  # duration in seconds

    # ------ Parameter sampling (vectorized) ------
    def sample_params_per_example(M: int):
        # log-uniform f0 in [fmin, fmax]
        lfmin, lfmax = math.log(fmin), math.log(fmax)
        f0 = torch.exp(lfmin + (lfmax - lfmin) * rand((M,)))  # [M]

        # random sweep direction s ∈ {+1, -1}
        s = torch.where(rand((M,)) < 0.5,
                        torch.tensor(1.0, device=device, dtype=dtype),
                        torch.tensor(-1.0, device=device, dtype=dtype))  # [M]

        # feasible octave spans to stay inside band
        up_max   = torch.log2(torch.tensor(fmax, device=device, dtype=dtype) / f0).clamp_min(0)   # [M]
        down_max = torch.log2(f0 / torch.tensor(fmin, device=device, dtype=dtype)).clamp_min(0)   # [M]

        desired = min_octaves + (max_octaves - min_octaves) * rand((M,))  # [M]
        feasible = torch.where(s > 0, up_max, down_max)
        delta = torch.minimum(desired, feasible).clamp_min(0.0)           # [M]
        valid_sweep = delta > 0

        r = torch.pow(2.0, s * delta)
        f1 = f0 * r
        k = (f1 - f0) / (N / sr if N > 0 else 1.0)  # Hz/s

        # amplitude and phase
        a_lo, a_hi = amp_db_range
        a_db = a_lo + (a_hi - a_lo) * rand((M,))
        A = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), a_db / 20.0)
        phi = 2 * math.pi * rand((M,))
        return f0, f1, k, A, phi, valid_sweep

    if C is None:
        # ---- Mono: one param set per chirp ----
        f0, f1, k, A, phi, valid_sweep = sample_params_per_example(n_chirps)  # [n_chirps]
        tn = t.unsqueeze(0)                      # [1, N]
        theta = 2*math.pi*(f0[:,None]*tn + 0.5*k[:,None]*tn*tn) + phi[:,None]  # [n_chirps, N]
        x_chirps = (A[:,None] * torch.cos(theta)).to(dtype)                    # [n_chirps, N]
    else:
        if same_chirp_across_channels:
            # ---- Multichannel: same chirp on all channels per example ----
            f0, f1, k, A, phi, valid_sweep = sample_params_per_example(n_chirps)  # [n_chirps]
            tn = t.unsqueeze(0)  # [1,N]
            theta_1c = 2*math.pi*(f0[:,None]*tn + 0.5*k[:,None]*tn*tn) + phi[:,None]  # [n_chirps,N]
            x_1c = (A[:,None] * torch.cos(theta_1c)).to(dtype)                       # [n_chirps,N]
            # broadcast across channels
            theta = theta_1c[:, None, :].expand(n_chirps, C, N).contiguous()         # [n_chirps,C,N]
            x_chirps = x_1c[:, None, :].expand(n_chirps, C, N).contiguous()          # [n_chirps,C,N]
        else:
            # ---- Multichannel: independent chirp per channel ----
            # Sample params per (example,channel)
            f0_c, f1_c, k_c, A_c, phi_c, valid_sweep_c = sample_params_per_example(n_chirps * C)
            # reshape to [n_chirps, C]
            f0 = f0_c.view(n_chirps, C); f1 = f1_c.view(n_chirps, C)
            k  = k_c.view(n_chirps, C);  A  = A_c.view(n_chirps, C)
            phi = phi_c.view(n_chirps, C); valid_sweep = valid_sweep_c.view(n_chirps, C)

            tn = t.view(1, 1, N)  # [1,1,N]
            theta = 2*math.pi*(f0.unsqueeze(-1)*tn + 0.5*k.unsqueeze(-1)*tn*tn) + phi.unsqueeze(-1)  # [n_chirps,C,N]
            x_chirps = (A.unsqueeze(-1) * torch.cos(theta)).to(dtype)                                 # [n_chirps,C,N]

    if add_dither and add_dither > 0:
        x_chirps = x_chirps + add_dither * randn_like(x_chirps)

    # Concatenate along batch dimension
    batch_aug = torch.cat([batch, x_chirps], dim=0).to(dtype)

    return batch_aug

class AutoencoderTrainingWrapper(pl.LightningModule):
    def __init__(
            self,
            autoencoder: AudioAutoencoder,
            sample_rate=48000,
            loss_config: Optional[dict] = None,
            eval_loss_config: Optional[dict] = None,
            optimizer_configs: Optional[dict] = None,
            lr: float = 1e-4,
            warmup_steps: int = 0,
            warmup_mode: Literal["adv", "full"] = "full",
            encoder_freeze_on_warmup: bool = False,
            use_ema: bool = True,
            ema_copy = None,
            force_input_mono = False,
            latent_mask_ratio = 0.0,
            teacher_model: Optional[AudioAutoencoder] = None,
            clip_grad_norm = 0.0,
            decoder_finetune = False,
            decoder_loss = False,
            num_synthetic_chirps: int = 0,
            stride_curriculum = None,
            log_every_n_steps: int = 10,
            tail_masking_max_patches: int = 0
    ):
        super().__init__()
        self.automatic_optimization = False
        #self.strict_loading = False
        self.autoencoder = autoencoder#torch.compile(autoencoder, mode="max-autotune",fullgraph=False)

        self.warmed_up = False if warmup_steps > 0 else True
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm

        self._staggered_logger = StaggeredLogger(every_n_steps=log_every_n_steps)

        self.force_input_mono = force_input_mono

        self.teacher_model = teacher_model
        self.decoder_finetune = decoder_finetune

        self.sample_rate = sample_rate

        self.decoder_loss = decoder_loss
        self.num_synthetic_chirps = num_synthetic_chirps

        self.stride_curriculum = stride_curriculum
        self.tail_masking_max_patches = tail_masking_max_patches


        if optimizer_configs is None:
            optimizer_configs ={
                "autoencoder": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (.8, .99)
                        }
                    }
                },
                "discriminator": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (.8, .99)
                        }
                    }
                }

            }

        self.optimizer_configs = optimizer_configs

        if loss_config is None:
            scales = [2048, 1024, 512, 256, 128, 64, 32]
            hop_sizes = []
            win_lengths = []
            overlap = 0.75
            for s in scales:
                hop_sizes.append(int(s * (1 - overlap)))
                win_lengths.append(s)

            loss_config = {
                "discriminator": {
                    "type": "encodec",
                    "config": {
                        "n_ffts": scales,
                        "hop_lengths": hop_sizes,
                        "win_lengths": win_lengths,
                        "filters": 32
                    },
                    "weights": {
                        "adversarial": 0.1,
                        "feature_matching": 5.0,
                    }
                },
                "spectral": {
                    "type": "mrstft",
                    "config": {
                        "fft_sizes": scales,
                        "hop_sizes": hop_sizes,
                        "win_lengths": win_lengths,
                        "perceptual_weighting": True
                    },
                    "weights": {
                        "mrstft": 1.0,
                    }
                },
                "time": {
                    "type": "l1",
                    "config": {},
                    "weights": {
                        "l1": 0.0,
                    }
                }
            }

        self.loss_config = loss_config

        # Spectral reconstruction loss
        if 'spectral' in loss_config:
            stft_loss_args = loss_config['spectral']['config']
            if self.autoencoder.out_channels == 2:
                self.sdstft = auraloss.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
                self.lrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            else:
                self.sdstft = auraloss.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

        self.use_contrastive_loss = True if 'contrastive' in loss_config else False

        if self.use_contrastive_loss:
            self.contrastive_warmup = loss_config['contrastive'].get('warmup', 10000)
            self.use_contrastive_text = loss_config['contrastive']['config'].get('use_text', False)
            if self.use_contrastive_text:
                gemma = T5GemmaConditioner(output_dim = 768)
                self.contrastive_text_conditioner = MultiConditioner({"prompt": gemma})
            self.critic_pretransform = WaveletPretransform(
                channels=self.autoencoder.in_channels,
                levels = 8
            )
            self.critic = LatentAudioCritic(audio_dim = self.critic_pretransform.encoded_channels,
                                latent_dim = self.autoencoder.latent_dim,
                                text_dim = 768 if self.use_contrastive_text else None,
                                transformer_dim = loss_config['contrastive']['config']['transformer_dim'],
                                depth = loss_config['contrastive']['config']['depth'],
                                dyt = loss_config['contrastive']['config'].get('dyt', True))
            set_requires_grad(self.critic, loss_config['contrastive']['config'].get('requires_grad', True))

        self.use_jepa_loss = True if 'jepa' in loss_config else False

        if self.use_jepa_loss:
            self.jepa_warmup = loss_config['jepa'].get('warmup', 10000)
            self.jepa = LatentJEPA(latent_dim = self.autoencoder.latent_dim,
                                transformer_dim = loss_config['jepa']['config']['transformer_dim'],
                                depth = loss_config['jepa']['config']['depth'],
                                mask_rate = loss_config['jepa']['config'].get('mask_rate', 0.4),
                                mask_block_size = loss_config['jepa']['config'].get('mask_block_size', 8),
                                dyt = loss_config['jepa']['config'].get('dyt', True))
            set_requires_grad(self.jepa, loss_config['jepa']['config'].get('requires_grad', True))

        self.use_generative_loss = True if 'generative' in loss_config else False

        if self.use_generative_loss:
            self.generative_warmup = loss_config['generative'].get('warmup', 10000)
            if loss_config['generative']['type'] == 'diffusion':
                self.generative_model = DiffusionTransformer(io_channels = self.autoencoder.latent_dim,
                                                             embed_dim = loss_config['generative']['config']['embed_dim'],
                                                             depth = loss_config['generative']['config']['depth'],
                                                             transformer_type = 'continuous_transformer',
                                                             timestep_features_dim = 512)
                #self.generative_model = torch.compile(self.generative_model, mode="max-autotune",fullgraph=False)

        self.use_disc = True if 'discriminator' in loss_config else False

        # Discriminator
        if self.use_disc:
            if loss_config['discriminator']['type'] == 'oobleck':
                self.discriminator = OobleckDiscriminator(**loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'encodec':
                self.discriminator = EncodecDiscriminator(in_channels=self.autoencoder.out_channels, **loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'dac':
                self.discriminator = DACGANLoss(channels=self.autoencoder.out_channels, sample_rate=sample_rate, **loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'big_vgan':
                self.discriminator = BigVGANDiscriminator(channels=self.autoencoder.out_channels, sample_rate=sample_rate,**loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'transformer':
                self.discriminator = MultiTransformerDiscriminator(in_channels=self.autoencoder.out_channels, sample_rate=sample_rate, **loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'hil':
                self.discriminator = HILDiscriminator(in_channels=self.autoencoder.out_channels, sample_rate=sample_rate, **loss_config['discriminator']['config'])
            else:
                raise ValueError(f"Unknown discriminator type: {loss_config['discriminator']['type']}")
            #self.discriminator = torch.compile(self.discriminator, mode="max-autotune",fullgraph=False)
        else:
            self.discriminator = None

        self.gen_loss_modules = []

        if self.use_generative_loss:
            diffusion_loss_weight = self.loss_config['generative']['weights']['diffusion']
            self.gen_loss_modules += [MSELoss(key_a='diffusion_output', key_b='diffusion_target',
                                             weight=diffusion_loss_weight,
                                             name='gen_step_diffusion_loss',
                                             decay = self.loss_config['generative'].get('decay', 1.0))]

        if self.use_jepa_loss:
            self.gen_loss_modules += [ValueLoss(key='jepa_loss',
                                             weight=self.loss_config['jepa']['weights']['jepa'],
                                             name='jepa_loss',
                                             decay=self.loss_config['jepa'].get('decay', 1.0))]
        if self.use_contrastive_loss:
            contrastive_loss_weight = self.loss_config['contrastive']['weights']['contrastive']
            self.gen_loss_modules += [ValueLoss(key='contrastive_loss',
                                             weight=contrastive_loss_weight,
                                             name='contrastive_loss',
                                             decay = self.loss_config['contrastive'].get('decay_contrastive', 1.0))]

        # Adversarial and feature matching losses
        if self.use_disc:
            disc_loss_decay = self.loss_config['discriminator'].get('decay', 1.0)
            self.gen_loss_modules += [
                ValueLoss(key='loss_adv', weight=self.loss_config['discriminator']['weights']['adversarial'], name='loss_adv', decay = disc_loss_decay),
                ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']['weights']['feature_matching'], name='feature_matching_loss', decay = disc_loss_decay),
            ]
        
        if "spectral" in loss_config:
            stft_loss_decay = self.loss_config['spectral'].get('decay', 1.0)
            if self.teacher_model is not None:
                # Distillation losses
                stft_loss_weight = self.loss_config['spectral']['weights']['mrstft'] * 0.25
                self.gen_loss_modules += [
                    AuralossLoss(self.sdstft, target_key = 'reals', input_key = 'decoded', name='mrstft_loss', weight=stft_loss_weight, decay = stft_loss_decay), # Reconstruction loss
                    AuralossLoss(self.sdstft, input_key = 'decoded', target_key = 'teacher_decoded', name='mrstft_loss_distill', weight=stft_loss_weight, decay = stft_loss_decay), # Distilled model's decoder is compatible with teacher's decoder
                    AuralossLoss(self.sdstft, target_key = 'reals', input_key = 'own_latents_teacher_decoded', name='mrstft_loss_own_latents_teacher', weight=stft_loss_weight, decay = stft_loss_decay), # Distilled model's encoder is compatible with teacher's decoder
                    AuralossLoss(self.sdstft, target_key = 'reals', input_key = 'teacher_latents_own_decoded', name='mrstft_loss_teacher_latents_own', weight=stft_loss_weight, decay = stft_loss_decay) # Teacher's encoder is compatible with distilled model's decoder
                ]

            else:
                # Reconstruction loss
                self.gen_loss_modules += [
                    AuralossLoss(self.sdstft, target_key = 'reals', input_key = 'decoded', name='mrstft_loss', weight=self.loss_config['spectral']['weights']['mrstft'], decay = stft_loss_decay),
                ]

                if self.autoencoder.out_channels == 2:
                    # Add left and right channel reconstruction losses in addition to the sum and difference
                    self.gen_loss_modules += [
                        AuralossLoss(self.lrstft, target_key = 'reals',  input_key = 'decoded', name='lr_stft_loss', weight=self.loss_config['spectral']['weights']['mrstft'], decay = stft_loss_decay)
                    ]

        # Direct latent distillation loss (L1/L2 between student and teacher latents)
        if self.teacher_model is not None and "latent_distillation" in loss_config:
            latent_distill_config = loss_config["latent_distillation"]
            latent_distill_weights = latent_distill_config.get("weights", {})
            latent_distill_decay = latent_distill_config.get("decay", 1.0)
            
            if latent_distill_weights.get("l1", 0.0) > 0.0:
                self.gen_loss_modules.append(L1Loss(key_a='latents', key_b='teacher_latents',
                                             weight=latent_distill_weights['l1'],
                                             name='latent_distill_l1',
                                             decay=latent_distill_decay))
            
            if latent_distill_weights.get("l2", 0.0) > 0.0:
                self.gen_loss_modules.append(MSELoss(key_a='latents', key_b='teacher_latents',
                                             weight=latent_distill_weights['l2'],
                                             name='latent_distill_l2',
                                             decay=latent_distill_decay))

        if "mrmel" in loss_config:
            mrmel_weight = loss_config["mrmel"]["weights"]["mrmel"]
            if mrmel_weight > 0:
                mrmel_config = loss_config["mrmel"]["config"]
                self.mrmel = MelSpectrogramLoss(sample_rate,
                    n_mels=mrmel_config["n_mels"],
                    window_lengths=mrmel_config["window_lengths"],
                    pow=mrmel_config["pow"],
                    log_weight=mrmel_config["log_weight"],
                    mag_weight=mrmel_config["mag_weight"],
                )
                self.gen_loss_modules.append(LossWithTarget(
                    self.mrmel, "reals", "decoded",
                    name="mrmel_loss", weight=mrmel_weight,
                ))

        if "hubert" in loss_config:
            hubert_weight = loss_config["hubert"]["weights"]["hubert"]
            if hubert_weight > 0:
                hubert_cfg = (
                    loss_config["hubert"]["config"]
                    if "config" in loss_config["hubert"] else dict())
                self.hubert = HubertLoss(weight=1.0, **hubert_cfg)

                self.gen_loss_modules.append(LossWithTarget(
                    self.hubert, target_key = "reals", input_key = "decoded",
                    name="hubert_loss", weight=hubert_weight,
                    decay = loss_config["hubert"].get("decay", 1.0)
                ))

        if "clap" in loss_config:
            clap_weight = loss_config["clap"]["weights"]["clap"]
            if clap_weight > 0:
                clap_cfg = (
                    loss_config["clap"]["config"]
                    if "config" in loss_config["clap"] else dict())
                self.clap_loss = CLAPLoss(sample_rate=sample_rate, weight=1.0, **clap_cfg)

                self.gen_loss_modules.append(LossWithTarget(
                    self.clap_loss, target_key = "reals", input_key = "decoded",
                    name="clap_loss", weight=clap_weight,
                    decay = loss_config["clap"].get("decay", 1.0)
                ))

        if "l1" in loss_config["time"]["weights"]:
            if self.loss_config['time']['weights']['l1'] > 0.0:
                self.gen_loss_modules.append(L1Loss(key_a='reals', key_b='decoded',
                                             weight=self.loss_config['time']['weights']['l1'],
                                             name='l1_time_loss',
                                             decay = self.loss_config['time'].get('decay', 1.0)))

        if "l2" in loss_config["time"]["weights"]:
            if self.loss_config['time']['weights']['l2'] > 0.0:
                self.gen_loss_modules.append(MSELoss(key_a='reals', key_b='decoded',
                                             weight=self.loss_config['time']['weights']['l2'],
                                             name='l2_time_loss',
                                             decay = self.loss_config['time'].get('decay', 1.0)))

        if "mmd" in loss_config["time"]["weights"]:
            if self.loss_config['time']['weights']['mmd'] > 0.0:
                self.gen_loss_modules.append(TimeDomainMMDLoss(key_a='reals', key_b='decoded',
                                             weight=self.loss_config['time']['weights']['mmd'],
                                             name='mmd_time_loss',
                                             decay = self.loss_config['time'].get('decay', 1.0)))

        if self.autoencoder.bottleneck is not None:
            self.gen_loss_modules += create_loss_modules_from_bottleneck(self.autoencoder.bottleneck, self.loss_config)

        if self.decoder_loss:
            self.gen_loss_modules.append(ValueLoss(key="decoder_gen_loss", weight = 2.0, name="decoder_gen_loss"))
        
        if 'semantic_regressors' in loss_config:
            weight = loss_config['semantic_regressors']['weights']['semantic_regressors_l1']
            decay = loss_config['semantic_regressors'].get('decay', 1.0)
            self.semantic_regressor_warmup = loss_config['semantic_regressors'].get('warmup', 10000)
            self.semantic_regressor_names = []
            self.semantic_transforms = torch.nn.ModuleList([])
            self.semantic_regressors = torch.nn.ModuleList([])
            self.chroma_centers = [1.0,5.0,9.0]
            center_widths = [1.0,1.5,1.0]
            center_weights = [0.035,0.05,0.2]
            self.spectrogram = TightSpectrogram(n_fft=2048 * 4, normalized = True, power = 1.0)
            for i,center in enumerate(self.chroma_centers):
                chroma_transform = torch.nn.Sequential(
                    MeanChannelLog1pTransform(),
                    torchaudio.prototype.transforms.ChromaScale(
                    sample_rate=sample_rate, n_chroma=128, n_freqs = 2048 * 2 + 1,
                    octwidth = center_widths[i], ctroct = center, norm = 1, base_c = True))
                self.semantic_transforms.append(chroma_transform)
                regressor = torch.nn.Conv1d(in_channels=self.autoencoder.latent_dim, out_channels=128, kernel_size=1, bias=True)
                self.semantic_regressors.append(regressor)
                name = f'chroma_band_{center}'
                self.semantic_regressor_names.append(name)
                self.gen_loss_modules.append(L1Loss(key_a= name + '_output',key_b = name + '_target', name = name + '_regressor_mse', weight = weight * center_weights[i], decay = decay))
            if self.autoencoder.out_channels == 2:
                ild_transform = ILDTransform(sample_rate=sample_rate, n_mels=32, n_fft=2048 * 4)
                self.semantic_transforms.append(ild_transform)
                ild_regressor = torch.nn.Conv1d(in_channels=self.autoencoder.latent_dim, out_channels=32, kernel_size=1, bias=True)
                self.semantic_regressors.append(ild_regressor)
                name = f'ild'
                self.semantic_regressor_names.append(name)
                self.gen_loss_modules.append(L1Loss(key_a= name + '_output',key_b =name + '_target',name = name + '_regressor_mse', weight = weight * 0.1, decay = decay))

        self.losses_gen = MultiLoss(self.gen_loss_modules)

        if self.use_disc:
            self.disc_loss_modules = [
                ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss'),
            ]
            if self.use_generative_loss:
                diffusion_loss_weight = self.loss_config['generative']['weights']['diffusion']
                self.disc_loss_modules += [MSELoss(key_a='diffusion_output', key_b='diffusion_target',
                                                weight=1.0,
                                                name='disc_step_diffusion_loss',
                                                decay = self.loss_config['generative'].get('decay', 1.0))]

            self.losses_disc = MultiLoss(self.disc_loss_modules)

        # Set up EMA for model weights
        self.autoencoder_ema = None

        self.use_ema = use_ema
        if self.use_ema:
            self.autoencoder_ema = EMA(
                self.autoencoder,
                ema_model=ema_copy,
                beta=0.999,
                power=3/4,
                update_every=1,  # Reduced from 1 to prevent throughput degradation
                update_after_step=2000  # Wait longer before starting EMA to stabilize training
            )

        self.latent_mask_ratio = latent_mask_ratio

        # evaluation losses & metrics
        self.eval_losses = torch.nn.ModuleDict()
        if eval_loss_config is not None:
            if "stft"in eval_loss_config:
                self.eval_losses["stft"] = auraloss.STFTLoss(**eval_loss_config["stft"])
            if "sisdr" in eval_loss_config:
                self.eval_losses["sisdr"] = auraloss.SISDRLoss(**eval_loss_config["sisdr"])
            if "mel" in eval_loss_config:
                self.eval_losses["mel"] = auraloss.MelSTFTLoss(
                    sample_rate, **eval_loss_config["mel"])

        self.validation_step_outputs = []

        self.latent_std = 1.0


    def configure_optimizers(self):
        if self.decoder_finetune:
            gen_params = list(self.autoencoder.decoder.parameters())
            if self.autoencoder.pretransform is not None:
                gen_params += list(self.autoencoder.pretransform.parameters())
        else:
            gen_params = list(self.autoencoder.parameters())
            if hasattr(self, "semantic_regressors"):
                gen_params += list(self.semantic_regressors.parameters())
            if self.use_generative_loss:
                gen_params += list(self.generative_model.parameters())
            if self.use_contrastive_loss:
                gen_params += list(self.critic.parameters())
            if self.use_jepa_loss:
                gen_params +=list(self.jepa.parameters())

        if self.use_disc:
            disc_params = list(self.discriminator.parameters())
            if self.use_generative_loss:
                disc_params += list(self.generative_model.parameters())
            opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], gen_params)
            opt_disc = create_optimizer_from_config(self.optimizer_configs['discriminator']['optimizer'], disc_params)
            if "scheduler" in self.optimizer_configs['autoencoder'] and "scheduler" in self.optimizer_configs['discriminator']:
                sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
                sched_disc = create_scheduler_from_config(self.optimizer_configs['discriminator']['scheduler'], opt_disc)
                return [opt_gen, opt_disc], [sched_gen, sched_disc]
            return [opt_gen, opt_disc]
        else:
            opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], gen_params)
            if "scheduler" in self.optimizer_configs['autoencoder']:
                sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
                return [opt_gen], [sched_gen]
            return [opt_gen]

    def forward(self, reals):
        latents, encoder_info = self.autoencoder.encode(reals, return_info=True)
        decoded = self.autoencoder.decode(latents)
        return decoded

    def on_train_epoch_end(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        reals, _ = batch

        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if len(reals.shape) == 2:
            reals = reals.unsqueeze(1)

        loss_info = {}

        loss_info["reals"] = reals

        encoder_input = reals

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        eval_model = self.autoencoder_ema.ema_model if self.autoencoder_ema is not None else self.autoencoder
        with torch.no_grad():
            latents, encoder_info = eval_model.encode(encoder_input, return_info=True)
            loss_info["latents"] = latents
            loss_info.update(encoder_info)
            decoded = eval_model.decode(latents)

            #Trim output to remove post-padding.
            decoded, reals = trim_to_shortest(decoded, reals)

        bs = reals.size(0)
        val_loss_dict = {}
        for name, fn in self.eval_losses.items():
            v = fn(decoded, reals)
            if name == "sisdr":
                v = -v
            if isinstance(v, torch.Tensor):
                v = v.detach()
            # store per-batch *sum* and *count* as tensors on the right device
            val_loss_dict[name] = {
                "sum": (v * bs).to(self.device) if not torch.is_tensor(v) else v * bs,
                "cnt": torch.tensor(bs, device=self.device, dtype=torch.float32),
            }
        self.validation_step_outputs.append(val_loss_dict)
        return val_loss_dict


    def on_validation_epoch_end(self):
        # initialize local totals
        keys = self.validation_step_outputs[0].keys() if self.validation_step_outputs else []
        totals = {k: torch.tensor(0.0, device=self.device) for k in keys}
        counts = {k: torch.tensor(0.0, device=self.device) for k in keys}
        
        # sum over this rank's batches
        for out in self.validation_step_outputs:
            for k in keys:
                totals[k] += out[k]["sum"]
                counts[k] += out[k]["cnt"]
        
        # all-reduce across ranks (sum, not mean)
        totals_all = {k: self.all_gather(totals[k]).sum() for k in keys}
        counts_all = {k: self.all_gather(counts[k]).sum() for k in keys}
        
        # global mean = global_sum / global_count
        for k in keys:
            val = (totals_all[k] / counts_all[k]).item()
            log_metric(self.logger, f"val/{k}", val, step=self.global_step)
        
        self.validation_step_outputs.clear()


    def training_step(self, batch, batch_idx):
        reals, json = batch
         
        log_dict = {}
        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if len(reals.shape) == 2:
            reals = reals.unsqueeze(1)

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        use_disc = (
            self.use_disc
            and self.global_step % 2
            # Check warmup mode and if it is time to use discriminator.
            and (
                (self.warmup_mode == "full" and self.warmed_up)
                or self.warmup_mode == "adv")
        )

        # Toggle discriminator requires_grad: only need disc param gradients on disc steps.
        # On gen steps, the disc forward is still needed for adversarial/FM loss, but
        # backward should only compute d(loss)/d(decoded), not d(loss)/d(disc_params).
        if self.use_disc:
            set_requires_grad(self.discriminator, bool(use_disc))

        loss_info = {}

        if self.num_synthetic_chirps > 0:
            reals = append_random_linear_chirps(reals, self.sample_rate, n_chirps = self.num_synthetic_chirps)

        # Rarely zero out entire examples so the model learns silence -> silence
        silence_mask = torch.rand(reals.shape[0], 1, 1, device=reals.device) < 1e-3
        reals = reals * ~silence_mask

        # Rarely zero out the last 1-N patches so the model learns to handle
        # the partial-chunk zero-padding added by _zero_pad_modulo_sequence
        if self.tail_masking_max_patches > 0 and self.autoencoder.pretransform is not None:
            patch_size = self.autoencoder.pretransform.downsampling_ratio
            tail_mask = torch.rand(reals.shape[0], device=reals.device) < 0.05
            if tail_mask.any():
                n_patches = torch.randint(1, self.tail_masking_max_patches + 1, (reals.shape[0],), device=reals.device)
                n_samples = n_patches * patch_size
                for i in torch.where(tail_mask)[0]:
                    reals[i, :, -n_samples[i]:] = 0

        encoder_input = reals

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        extra_kwargs = {}

        if not self.decoder_finetune and self.stride_curriculum is not None:
            #############
            index = (self.global_step // 2000) % (len(self.stride_curriculum))
            if (random.randint(0,8) != 0):
                extra_kwargs['override_stride'] = None
            else:
                extra_kwargs['override_stride'] = [max(int(x * self.stride_curriculum[index]),1) for x in self.autoencoder.encoder.strides]
            #############

        encoder_grad_disabled = use_disc or self.decoder_finetune or (self.warmed_up and self.encoder_freeze_on_warmup)

        with torch.no_grad() if encoder_grad_disabled else nullcontext():
            if self.decoder_loss:
                latents, encoder_info, encoder_transformer_input = self.autoencoder.encode(encoder_input, return_info=True, return_pretransform=True, **extra_kwargs)
            else:
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True, **extra_kwargs)

        loss_info["latents"] = latents


        loss_info.update(encoder_info)
        latent_std = latents.std().detach().item()
        
        if not encoder_grad_disabled:
            self.latent_std = 0.99 * self.latent_std + 0.01*latent_std

            if hasattr(self, "semantic_transforms"):
                spectrogram = self.spectrogram(encoder_input)
                for i, transform in enumerate(self.semantic_transforms):
                    target = transform(spectrogram)
                    if self.global_step < self.semantic_regressor_warmup:
                        output = self.semantic_regressors[i](latents.detach())
                    else:
                        output = self.semantic_regressors[i](latents)
                    if output.shape[-1] != target.shape[-1]:
                        target = torch.nn.functional.interpolate(target, size=output.shape[-1], mode='linear')
                    loss_info[f'{self.semantic_regressor_names[i]}_target'] = target
                    loss_info[f'{self.semantic_regressor_names[i]}_output'] = output

            if self.use_contrastive_loss:
                critic_audio = self.critic_pretransform.encode(reals)
                text = None
                if self.use_contrastive_text:
                    with torch.no_grad():
                        text = self.contrastive_text_conditioner(json, reals.device)['prompt'][0]
                if self.global_step < self.contrastive_warmup:
                    loss_info["contrastive_loss"] = self.critic.loss(critic_audio, latents.detach(), text=text)
                else:
                    loss_info["contrastive_loss"] = self.critic.loss(critic_audio, latents, text=text)
            
            if self.use_jepa_loss:
                if self.global_step < self.jepa_warmup:
                    loss_info["jepa_loss"] = self.jepa.loss(latents.detach())
                else:
                    loss_info["jepa_loss"] = self.jepa.loss(latents)

            if self.use_generative_loss:
                #t = torch.rand(reals.shape[0], device=self.device)
                t = 1 - truncated_logistic_normal_rescaled(reals.shape[0]).to(self.device)
                alphas, sigmas = 1-t, t
                alphas = alphas[:, None, None]
                sigmas = sigmas[:, None, None]
                if self.global_step < self.generative_warmup:
                    scaled_latents = latents.detach()
                else:
                    scaled_latents = latents 
                noise = torch.randn_like(scaled_latents) * self.latent_std
                noised_inputs = scaled_latents * alphas + noise * sigmas
                loss_info["diffusion_target"] = (noise - scaled_latents) 
                loss_info["diffusion_output"] = self.generative_model(noised_inputs, t) 
                log_dict['train/diffusion_loss_zero'] = torch.nn.functional.mse_loss(loss_info["diffusion_target"], torch.zeros_like(loss_info["diffusion_target"])).detach()
            

        # Encode with teacher model for distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_latents = self.teacher_model.encode(encoder_input, return_info=False)
                loss_info['teacher_latents'] = teacher_latents

        # Optionally mask out some latents for noise resistance
        if self.latent_mask_ratio > 0.0:
            mask = torch.rand_like(latents) < self.latent_mask_ratio
            latents = torch.where(mask, torch.zeros_like(latents), latents)

        with torch.no_grad() if use_disc else nullcontext():
            if self.decoder_loss:
                log_dict['train/pretransform_std'] = encoder_transformer_input.std().detach()
                decoded, decoder_gen_loss = self.autoencoder.decode(latents, generative_target = encoder_transformer_input.detach().transpose(-1,-2), return_loss = True, **extra_kwargs)
                log_dict.update(self.autoencoder.decoder._last_loss_dict)
                loss_info["decoder_gen_loss"] = decoder_gen_loss
            else:
                decoded = self.autoencoder.decode(latents, **extra_kwargs)
                
        decoded, reals = zero_pad_to_longest(decoded, reals)

        loss_info["decoded"] = decoded
        loss_info["reals"] = reals

        if self.autoencoder.out_channels == 2:
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = reals[:, 0:1, :]
            loss_info["reals_right"] = reals[:, 1:2, :]

        # Distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_decoded = self.teacher_model.decode(teacher_latents)
                own_latents_teacher_decoded = self.teacher_model.decode(latents) #Distilled model's latents decoded by teacher
                teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents) #Teacher's latents decoded by distilled model

                # Trim to match reals length (may be longer due to chunk padding)
                teacher_decoded, _ = trim_to_shortest(teacher_decoded, reals)
                own_latents_teacher_decoded, _ = trim_to_shortest(own_latents_teacher_decoded, reals)
                teacher_latents_own_decoded, _ = trim_to_shortest(teacher_latents_own_decoded, reals)

                loss_info['teacher_decoded'] = teacher_decoded
                loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
                loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded

        if self.use_disc:
            if self.warmed_up:
                loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals=reals, fakes=decoded)
                
                # Cross-term adversarial losses for distillation
                # These ensure latent compatibility at a perceptual level
                if self.teacher_model is not None:
                    # Trim cross-decoded outputs to match reals (already trimmed, but for safety)
                    teacher_decoded_padded, _ = trim_to_shortest(loss_info['teacher_decoded'], reals)
                    own_latents_teacher_decoded_padded, _ = trim_to_shortest(loss_info['own_latents_teacher_decoded'], reals)
                    teacher_latents_own_decoded_padded, _ = trim_to_shortest(loss_info['teacher_latents_own_decoded'], reals)
                    
                    # Student latents decoded by teacher should look real
                    loss_dis_cross1, loss_adv_cross1, fm_cross1 = self.discriminator.loss(reals=reals, fakes=own_latents_teacher_decoded_padded)
                    # Teacher latents decoded by student should look real  
                    loss_dis_cross2, loss_adv_cross2, fm_cross2 = self.discriminator.loss(reals=reals, fakes=teacher_latents_own_decoded_padded)
                    # Student decoded should match teacher decoded (use teacher as "real")
                    loss_dis_cross3, loss_adv_cross3, fm_cross3 = self.discriminator.loss(reals=teacher_decoded_padded, fakes=decoded)
                    
                    # Aggregate cross-term losses
                    cross_weight = 0.25  # Weight each cross-term relative to main loss
                    loss_dis = loss_dis + cross_weight * (loss_dis_cross1 + loss_dis_cross2 + loss_dis_cross3)
                    loss_adv = loss_adv + cross_weight * (loss_adv_cross1 + loss_adv_cross2 + loss_adv_cross3)
                    feature_matching_distance = feature_matching_distance + cross_weight * (fm_cross1 + fm_cross2 + fm_cross3)
                    
                    log_dict['train/loss_adv_cross_own_latents_teacher'] = loss_adv_cross1.detach()
                    log_dict['train/loss_adv_cross_teacher_latents_own'] = loss_adv_cross2.detach()
                    log_dict['train/loss_adv_cross_decoded_vs_teacher'] = loss_adv_cross3.detach()
            else:
                loss_adv = torch.tensor(0.).to(reals)
                feature_matching_distance = torch.tensor(0.).to(reals)

                if self.warmup_mode == "adv":
                    loss_dis, _, _ = self.discriminator.loss(reals=reals, fakes=decoded)
                else:
                    loss_dis = torch.tensor(0.0).to(reals)

            loss_info["loss_dis"] = loss_dis
            loss_info["loss_adv"] = loss_adv
            loss_info["feature_matching_distance"] = feature_matching_distance

        opt_gen = None
        opt_disc = None

        if self.use_disc:
            opt_gen, opt_disc = self.optimizers()
        else:
            opt_gen = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        sched_gen = None
        sched_disc = None

        scaler = getattr(self.trainer.precision_plugin, "scaler", None)

        if lr_schedulers is not None:
            if self.use_disc:
                sched_gen, sched_disc = lr_schedulers
            else:
                sched_gen = lr_schedulers


        if use_disc:
            loss, losses = self.losses_disc(loss_info)

            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']

            opt_disc.zero_grad()
            self.manual_backward(loss)
            if scaler is not None:
                scale = scaler.get_scale()
                log_dict['grads/disc_grad_scale'] = scale
            if self.clip_grad_norm > 0.0:
                if scaler is not None:
                    clip_norm_value = self.clip_grad_norm * scale
                else:
                    clip_norm_value = self.clip_grad_norm
                self.clip_gradients(opt_disc, gradient_clip_val=clip_norm_value, gradient_clip_algorithm="norm")

            if self.global_step % 10 == 0:
                rho_pred_disc = calc_update_to_weight_ratio(self.discriminator, opt_disc.param_groups[0]['lr'], scaler)
                log_dict['grads/disc_rho_pred'] = rho_pred_disc
            #for name, p in self.discriminator.named_parameters():
            #    if p.grad is not None and not torch.isfinite(p.grad).all():
            #        print("NaN grad in", name)

            
            opt_disc.step()
            if sched_disc is not None:
                # sched step every step
                sched_disc.step()

        # Train the generator
        else:

            loss, losses = self.losses_gen(loss_info)

            opt_gen.zero_grad()
            self.manual_backward(loss)
            if scaler is not None:
                scale = scaler.get_scale()
                log_dict['grads/gen_grad_scale'] = scale
            if self.clip_grad_norm > 0.0:
                if scaler is not None:
                    clip_norm_value = self.clip_grad_norm * scale
                else:
                    clip_norm_value = self.clip_grad_norm
                self.clip_gradients(opt_gen, gradient_clip_val=clip_norm_value, gradient_clip_algorithm="norm")
            if self.global_step % 10 == 0:
                rho_pred_gen = calc_update_to_weight_ratio(self.autoencoder, opt_gen.param_groups[0]['lr'], scaler)
                log_dict['grads/gen_rho_pred'] = rho_pred_gen
            #for name, p in self.autoencoder.named_parameters():
            #    if p.grad is not None and not torch.isfinite(p.grad).all():
            #        print("NaN grad in", name)
            opt_gen.step()

            # Update EMA less frequently to reduce computational overhead
            if self.use_ema and (self.global_step + 1) % 10 == 0:
                self.autoencoder_ema.update()

            if sched_gen is not None:
                # scheduler step every step
                sched_gen.step()

            log_dict['train/loss'] = loss.detach()
            log_dict['train/latent_std'] = latent_std  # already a Python float from .item() above
            log_dict['train/latent_mean'] = latents.mean().detach()
            log_dict['train/data_std'] = data_std.detach()
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach()

        self._staggered_logger.log(log_dict, self)
        
        return loss

    def export_model(self, path, use_safetensors=False):
        if self.autoencoder_ema is not None:
            model = self.autoencoder_ema.ema_model
        else:
            model = self.autoencoder

        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)

class AutoencoderDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        sample_size=65536,
        sample_rate=44100,
        max_demos = 8
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_dl = demo_dl
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.max_demos = max_demos


    @rank_zero_only
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if (self.last_demo_step >= 0 and trainer.global_step % self.demo_every != 0) or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step
        module.eval()

        try:
            demo_iter = iter(self.demo_dl)
            demo_reals, json  = next(demo_iter)

            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]

            # Limit the number of demo samples
            if demo_reals.shape[0] > self.max_demos:
                demo_reals = demo_reals[:self.max_demos,...]

            demo_reals = append_random_linear_chirps(demo_reals, self.sample_rate, n_chirps = 1)

            encoder_input = demo_reals
            encoder_input = encoder_input.to(module.device)

            if module.force_input_mono:
                encoder_input = encoder_input.mean(dim=1, keepdim=True)

            demo_reals = demo_reals.to(module.device)

            log_latent_sensitivity = False
            if log_latent_sensitivity:
                test_latents_delta = torch.zeros_like(latents, requires_grad=True)
                if module.use_ema:
                    latents = module.autoencoder_ema.ema_model.encode(encoder_input)
                    fakes = module.autoencoder_ema.ema_model.decode(latents + test_latents_delta)
                else:
                    latents = module.autoencoder.encode(encoder_input)
                    fakes = module.autoencoder.decode(latents + test_latents_delta)
                output_sum = fakes.sum()
                test_latents_delta.retain_grad()
                output_sum.backward()
                grads = test_latents_delta.grad.detach().abs()
                log_image(trainer.logger, 'latent_sensitivity', tokens_spectrogram_image(grads.log10(), title = 'Latent Sensitivity', symmetric = False), step=trainer.global_step)
                opts = module.optimizers()
                opts[0].zero_grad()
                opts[1].zero_grad()
            else:
                with torch.no_grad():
                    if module.use_ema:
                        latents = module.autoencoder_ema.ema_model.encode(encoder_input)
                        fakes = module.autoencoder_ema.ema_model.decode(latents)
                    else:
                        latents = module.autoencoder.encode(encoder_input)
                        fakes = module.autoencoder.decode(latents)

            #Trim output to remove post-padding.
            fakes, demo_reals = trim_to_shortest(fakes.detach(), demo_reals)
            log_dict = {}

            if module.discriminator is not None:
                window = torch.kaiser_window(512).to(fakes.device)
                fakes_stft = torch.stft(fold_channels_into_batch(fakes), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                fakes_stft.requires_grad = True
                fakes_signal = unfold_channels_from_batch(torch.istft(fakes_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), fakes.shape[1])
                real_stft = torch.stft(fold_channels_into_batch(demo_reals), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                reals_signal = unfold_channels_from_batch(torch.istft(real_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), demo_reals.shape[1])
                _, loss, _ = module.discriminator.loss(reals_signal,fakes_signal)
                fakes_stft.retain_grad()
                loss.backward()
                grads = unfold_channels_from_batch(fakes_stft.grad.detach().abs(),fakes.shape[1])
                log_image(trainer.logger, 'discriminator_sensitivity', tokens_spectrogram_image(grads.mean(dim=1).log10(), title = 'Discriminator Sensitivity', symmetric = False), step=trainer.global_step)
                opts = module.optimizers()
                opts[0].zero_grad()
                opts[1].zero_grad()

            if hasattr(module, "semantic_regressors"):
                spectrogram = module.spectrogram(demo_reals)
                for transform, regressor, name in zip(module.semantic_transforms, module.semantic_regressors, module.semantic_regressor_names):
                    target = transform(spectrogram)
                    output = regressor(latents)
                    if output.shape[-1] != target.shape[-1]:
                        target = torch.nn.functional.interpolate(target, size=output.shape[-1], mode='linear')
                    log_image(trainer.logger, f'{name}_target', tokens_spectrogram_image(target.detach().cpu(), title = f'{name} Target', symmetric = False), step=trainer.global_step)
                    log_image(trainer.logger, f'{name}_output', tokens_spectrogram_image(output.detach().cpu(), title = f'{name} Regressor Output', symmetric = False), step=trainer.global_step)


            #Interleave reals and fakes
            reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')
            # Put the demos together
            reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')
            
            try:
                data_dir = os.path.join(
                    trainer.logger.save_dir, logger_project_name(trainer.logger),
                    trainer.logger.experiment.id, "media")
                os.makedirs(data_dir, exist_ok=True)
                filename = os.path.join(data_dir, f'recon_{trainer.global_step:08}.wav')
            except:
                filename = f'recon_{trainer.global_step:08}.wav'

            reals_fakes = reals_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reals_fakes, self.sample_rate)

            log_audio(trainer.logger, 'recon', filename, self.sample_rate, step=trainer.global_step)
            os.remove(filename)
            log_image(trainer.logger, 'embeddings_spec',
                tokens_spectrogram_image(latents, symmetric = False), step=trainer.global_step)
            log_image(trainer.logger, 'recon_melspec_left',
                audio_spectrogram_image(reals_fakes), step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()

def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []

    if isinstance(bottleneck, VAEBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        try:
            kl_weight = loss_config['bottleneck']['weights']['kl']
        except:
            kl_weight = 1e-6

        kl_loss = ValueLoss(key='kl', weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)

    if isinstance(bottleneck, RVQBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        quantizer_loss = ValueLoss(key='quantizer_loss', weight=1.0, name='quantizer_loss')
        losses.append(quantizer_loss)

    if isinstance(bottleneck, DACRVQBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck):
        codebook_loss = ValueLoss(key='vq/codebook_loss', weight=1.0, name='codebook_loss')
        commitment_loss = ValueLoss(key='vq/commitment_loss', weight=0.25, name='commitment_loss')
        losses.append(codebook_loss)
        losses.append(commitment_loss)

    if isinstance(bottleneck, WassersteinBottleneck):
        try:
            mmd_weight = loss_config['bottleneck']['weights']['mmd']
        except:
            mmd_weight = 100

        mmd_loss = ValueLoss(key='mmd', weight=mmd_weight, name='mmd_loss')
        losses.append(mmd_loss)

    if isinstance(bottleneck, SoftNormBottleneck):
        try:
            softnorm_weight = loss_config['bottleneck']['weights']['softnorm']
        except:
            softnorm_weight = 1e-5

        softnorm_loss = ValueLoss(key='softnorm_loss', weight=softnorm_weight, name='softnorm_loss')
        losses.append(softnorm_loss)

    return losses

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchaudio
import numpy as np
import typing as tp

import scipy.signal

from typing import List, Tuple
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import Resample
from torchaudio.prototype.transforms import ChromaSpectrogram
from functools import reduce
from einops import rearrange


from .autoencoders import fold_channels_into_batch, unfold_channels_from_batch, checkpoint, TAAEEncoder, TransformerResamplingBlock
from .transforms import PQMF, TightSpectrogram
from .pretransforms import ComplexSTFTPretransform, PatchedPretransform, WaveletPretransform, HILPQMFPretransform
from .transformer import TransformerBlock
from .psychoacoustics import PsychoacousticStereoNoise

from .wavelets import WaveletEncode1d

def get_hinge_losses(score_real, score_fake):
    gen_loss = -score_fake.mean()
    dis_loss = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()
    return dis_loss, gen_loss

def get_sigmoid_relgan_losses(score_real, score_fake):
    diff = 0.5 * (score_fake - score_real)
    dis_loss = 2.0 * F.sigmoid(diff).mean()
    gen_loss = 2.0 * F.sigmoid(-diff).mean()
    return dis_loss, gen_loss

def get_relativistic_losses(score_real, score_fake):
    # Compute difference between real and fake scores
    diff = score_real - score_fake
    dis_loss = F.softplus(-diff).mean()
    gen_loss = F.softplus(diff).mean()
    return dis_loss, gen_loss

def create_blocked_mask(x, block_size, num_blocks):
    shape = x.shape
    mask = torch.zeros_like(x)
    for _ in range(num_blocks):
        block_start = torch.randint(0, shape[-1] - block_size, (shape[0],))
        block_end = block_start + block_size
        for i in range(shape[0]):
            mask[i,:, block_start[i]:block_end[i]] = 1
    return mask.bool()

def huber_loss(x,y):
    delta = 1.0
    diff = x - y
    loss = torch.where(torch.abs(diff) < delta, 0.5 * diff ** 2, delta * (torch.abs(diff) - 0.5 * delta))
    return loss.mean()

class LatentJEPA(nn.Module):
    def __init__(self, latent_dim, transformer_dim, depth, mask_rate = 0.4, mask_block_size = 8, dyt = True, *args, **kwargs):
        super().__init__()
        self.mask_rate = 0.4
        self.mask_block_size = 8
        layers = []
        layers.append(nn.Linear(latent_dim, transformer_dim))
        for i in range(depth):
            block = TransformerBlock(transformer_dim, 
                         dim_heads = 64, 
                         causal = False,  
                         zero_init_branch_outputs = True, 
                         conformer = False, 
                         layer_scale = False, 
                         add_rope = True,
                         norm_type = "dyt" if dyt else "rms_norm",
                         attn_kwargs={'qk_norm': "dyt" if dyt else "rms", "differential": False, "feat_scale": True}, 
                         ff_kwargs={'mult': 2, 'no_bias': False, "sinusoidal": False})
            layers.append(block)
        layers.append(nn.Linear(transformer_dim, latent_dim))
        self.layers = nn.Sequential(*layers)
        self.mask_token = nn.Parameter(torch.randn(1,latent_dim,1)*1e-2)
    
    def forward(self, x):
        return self.layers(x.transpose(-1,-2)).transpose(-1,-2)
    
    def loss(self, x):
        mask = create_blocked_mask(x, block_size = self.mask_block_size, num_blocks = int(self.mask_rate * x.shape[-1] / self.mask_block_size))
        masked_x = torch.where(mask, self.mask_token, x)
        pred = self.forward(masked_x)
        loss = huber_loss(pred[mask], x[mask].detach())
        return loss


class LatentAudioCritic(nn.Module):
    def __init__(self, audio_dim, latent_dim, text_dim = None, transformer_dim = 512, depth = 4, seq_mask_rate = 0.4, feature_mask_rate = 0.35, latent_mask_rate = 0.1, volume_aug_depth = 0.2, margin = 0.5, dyt = False,*args, **kwargs): #feature_mask_rate = 0.25, margin = 0.5/0.75
        super().__init__()

        self.seq_mask_rate = seq_mask_rate
        self.feature_mask_rate = feature_mask_rate
        self.latent_mask_rate = latent_mask_rate
        self.volume_aug_depth = volume_aug_depth
        
        self.margin = margin

        self.audio_mapping = nn.Linear(audio_dim, transformer_dim)
        self.latent_mapping = nn.Linear(latent_dim, transformer_dim)
        if text_dim is not None:
            self.use_text = True
            self.text_mapping = nn.Linear(text_dim, transformer_dim)
        else:
            self.use_text = False
        self.critic_mapping = nn.utils.parametrizations.weight_norm(nn.Linear(transformer_dim, 1))

        self.critic_token = nn.Parameter(1e-2 * torch.randn(1, 1, transformer_dim))

        discriminators = []
        for _ in range(depth):
            block = TransformerBlock(transformer_dim, 
                                     dim_heads = 64, 
                                     causal = False,  
                                     zero_init_branch_outputs = True, 
                                     conformer = False, 
                                     layer_scale = False, 
                                     add_rope = True, 
                                     norm_type = "dyt" if dyt else "rms_norm",
                                     attn_kwargs={'qk_norm': "dyt" if dyt else "rms", "differential": False, "feat_scale": True}, 
                                     ff_kwargs={'mult': 2, 'no_bias': False, "sinusoidal": False})
            discriminators.append(block)
        self.discriminators = nn.ModuleList(discriminators)

    def _mask_audio_emb(self,audio_emb):
        mask_seq = torch.rand(audio_emb.shape[0], audio_emb.shape[1],1, device = audio_emb.device) < self.seq_mask_rate
        mask_seq = mask_seq.expand(audio_emb.shape)
        audio_emb = audio_emb.masked_fill(mask_seq, 0.0)
        audio_emb = torch.nn.functional.dropout(audio_emb, p = self.feature_mask_rate)
        return audio_emb

    def _mask_latent_emb(self, latent_emb):
        latent_emb = torch.nn.functional.dropout(latent_emb, p = self.latent_mask_rate) + torch.randn_like(latent_emb)*1e-2
        return latent_emb

    def forward(self, audio, latent, text = None, masking = True):
        # Map audio and latent inputs to transformer dimension
        if masking:
            audio = audio * (1 + self.volume_aug_depth * torch.rand(1, device = audio.device))
        audio_emb = self.audio_mapping(audio.transpose(-1,-2))
        latent_emb = self.latent_mapping(latent.transpose(-1,-2))
        if self.use_text:
            text_emb = self.text_mapping(text)
        else:
            text_emb = torch.zeros(audio_emb.shape[0], 0, audio_emb.shape[-1], device = audio_emb.device)

        if masking:
            audio_emb = self._mask_audio_emb(audio_emb)
            latent_emb = self._mask_latent_emb(latent_emb)

        # Concatenate along sequence dimension
        combined = torch.cat([audio_emb, text_emb, latent_emb], dim=1)

        # Add critic token
        batch_size = combined.size(0)
        critic_token = self.critic_token.expand(batch_size, -1, -1)
        combined = torch.cat([combined, critic_token], dim=1)

        # Pass through transformer blocks
        for block in self.discriminators:
            combined = block(combined)

        # Extract critic token output
        critic_output = combined[:, -1, :]
        critic_output = torch.nn.functional.normalize(critic_output, dim = -1, eps = 1e-5)
        score = self.critic_mapping(critic_output)
        return score, combined[:, :audio_emb.shape[1],:]

    def loss(self, audio, latent, text = None, checkpointing = True):
        if checkpointing:
            logits_pos, _ = checkpoint(self.forward, audio, latent, text)
        else:
            logits_pos, _ = self.forward(audio, latent, text)
        loss = 0
        target_negatives = 1024
        negatives_per_example = audio.shape[0] * 3
        examples = min(target_negatives // negatives_per_example, audio.shape[0] - 1)
        for i in range(examples):
            shift = i + 1
            audio_rotated = torch.roll(audio, shifts=shift, dims=0)
            if checkpointing:
                logits_neg_audio, _ = checkpoint(self.forward, audio_rotated, latent, text)
            else:
                logits_neg_audio, _ = self.forward(audio_rotated, latent, text)
            loss += self._calculate_loss(logits_pos, logits_neg_audio)
            if text is not None:
                text_rotated = torch.roll(text, shifts=-shift, dims=0)
                if checkpointing:
                    logits_neg_text, _ = checkpoint(self.forward, audio, latent, text_rotated)
                else:
                    logits_neg_text, _ = self.forward(audio, latent, text_rotated)
                loss += self._calculate_loss(logits_pos, logits_neg_text)
                if checkpointing:
                    logits_neg_both, _ = checkpoint(self.forward, audio_rotated, latent, text_rotated)
                else:
                    logits_neg_both, _ = self.forward(audio_rotated, latent, text_rotated)
                loss += self._calculate_loss(logits_pos, logits_neg_both)
        loss = loss / examples
        if text is not None:
            loss = loss / 3
        if torch.isnan(loss):
            loss = torch.tensor(0., device=loss.device)
        return loss

    def _calculate_loss(self, logits_pos, logits_neg):
        diff = (logits_pos - logits_neg).clamp(max = 10)
        loss = torch.nn.functional.softplus(self.margin - diff).mean()
        return loss


class EncodecDiscriminator(nn.Module):
    def __init__(self, normalize_losses=False, loss_type: tp.Literal["hinge", "rpgan"]="hinge", *args, **kwargs):
        super().__init__()
        from .encodec import MultiScaleSTFTDiscriminator
        self.discriminators = MultiScaleSTFTDiscriminator(*args, **kwargs)
        self.normalize_losses = normalize_losses
        self.fm_reduction = (lambda x, y: abs(x - y).mean()/(abs(x).mean() + 1e-3)) if normalize_losses else (lambda x, y: abs(x - y).mean())
        self.loss_type = loss_type

    def forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def loss(self, reals, fakes):
        feature_matching_distance = torch.tensor(0., device=reals.device)
        dis_loss = torch.tensor(0., device=reals.device)
        adv_loss = torch.tensor(0., device=reals.device)

        logits_true, feature_true = self.forward(reals)
        logits_fake, feature_fake = self.forward(fakes)

        # Compute per-scale losses
        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):
            feature_matching_distance = feature_matching_distance + sum(
                map(
                    self.fm_reduction,
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            if self.loss_type == "hinge":
                _dis, _adv = get_hinge_losses(logits_true[i], logits_fake[i])
            else:  # rpgan
                _dis, _adv = get_relativistic_losses(logits_true[i], logits_fake[i])

            dis_loss = dis_loss + _dis 
            adv_loss = adv_loss + _adv

        num_scales = len(logits_true)

        return dis_loss / num_scales, adv_loss / num_scales, feature_matching_distance / num_scales


# Discriminators from oobleck

IndividualDiscriminatorOut = tp.Tuple[torch.Tensor, tp.Sequence[torch.Tensor]]

TensorDict = tp.Dict[str, torch.Tensor]

class SharedDiscriminatorConvNet(nn.Module):

    def __init__(
        self,
        in_size: int,
        convolution: tp.Union[nn.Conv1d, nn.Conv2d],
        out_size: int = 1,
        capacity: int = 32,
        n_layers: int = 4,
        kernel_size: int = 15,
        stride: int = 4,
        activation: tp.Callable[[], nn.Module] = lambda: nn.SiLU(),
        normalization: tp.Callable[[nn.Module], nn.Module] = torch.nn.utils.weight_norm,
    ) -> None:
        super().__init__()
        channels = [in_size]
        channels += list(capacity * 2**np.arange(n_layers))

        if isinstance(stride, int):
            stride = n_layers * [stride]

        net = []
        for i in range(n_layers):
            if isinstance(kernel_size, int):
                pad = kernel_size // 2
                s = stride[i]
            else:
                pad = kernel_size[0] // 2
                s = (stride[i], 1)

            net.append(
                normalization(
                    convolution(
                        channels[i],
                        channels[i + 1],
                        kernel_size,
                        stride=s,
                        padding=pad,
                    )))
            net.append(activation())

        net.append(convolution(channels[-1], out_size, 1))

        self.net = nn.ModuleList(net)

    def forward(self, x) -> IndividualDiscriminatorOut:
        features = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.modules.conv._ConvNd):
                features.append(x)
        score = x.reshape(x.shape[0], -1).mean(-1)
        return score, features


class MultiScaleDiscriminator(nn.Module):

    def __init__(self,
                in_channels: int,
                n_scales: int,
                **conv_kwargs) -> None:
        super().__init__()
        layers = []
        for _ in range(n_scales):
            layers.append(SharedDiscriminatorConvNet(in_channels, nn.Conv1d, **conv_kwargs))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> IndividualDiscriminatorOut:
        score = 0
        features = []
        for layer in self.layers:
            s, f = layer(x)
            score = score + s
            features.extend(f)
            x = nn.functional.avg_pool1d(x, 2)
        return score, features

class MultiPeriodDiscriminator(nn.Module):

    def __init__(self,
                 in_channels: int,
                 periods: tp.Sequence[int],
                 **conv_kwargs) -> None:
        super().__init__()
        layers = []
        self.periods = periods

        for _ in periods:
            layers.append(SharedDiscriminatorConvNet(in_channels, nn.Conv2d, **conv_kwargs))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> IndividualDiscriminatorOut:
        score = 0
        features = []
        for layer, n in zip(self.layers, self.periods):
            s, f = layer(self.fold(x, n))
            score = score + s
            features.extend(f)
        return score, features

    def fold(self, x: torch.Tensor, n: int) -> torch.Tensor:
        pad = (n - (x.shape[-1] % n)) % n
        x = nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:2], -1, n)


class MultiDiscriminator(nn.Module):
    """
    Individual discriminators should take a single tensor as input (NxB C T) and
    return a tuple composed of a score tensor (NxB) and a Sequence of Features
    Sequence[NxB C' T'].
    """

    def __init__(self, discriminator_list: tp.Sequence[nn.Module],
                 keys: tp.Sequence[str]) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(discriminator_list)
        self.keys = keys

    def unpack_tensor_to_dict(self, features: torch.Tensor) -> TensorDict:
        features = features.chunk(len(self.keys), 0)
        return {k: features[i] for i, k in enumerate(self.keys)}

    @staticmethod
    def concat_dicts(dict_a, dict_b):
        out_dict = {}
        keys = set(list(dict_a.keys()) + list(dict_b.keys()))
        for k in keys:
            out_dict[k] = []
            if k in dict_a:
                if isinstance(dict_a[k], list):
                    out_dict[k].extend(dict_a[k])
                else:
                    out_dict[k].append(dict_a[k])
            if k in dict_b:
                if isinstance(dict_b[k], list):
                    out_dict[k].extend(dict_b[k])
                else:
                    out_dict[k].append(dict_b[k])
        return out_dict

    @staticmethod
    def sum_dicts(dict_a, dict_b):
        out_dict = {}
        keys = set(list(dict_a.keys()) + list(dict_b.keys()))
        for k in keys:
            out_dict[k] = 0.
            if k in dict_a:
                out_dict[k] = out_dict[k] + dict_a[k]
            if k in dict_b:
                out_dict[k] = out_dict[k] + dict_b[k]
        return out_dict

    def forward(self, inputs: TensorDict) -> TensorDict:
        discriminator_input = torch.cat([inputs[k] for k in self.keys], 0)
        all_scores = []
        all_features = []

        for discriminator in self.discriminators:
            score, features = discriminator(discriminator_input)
            scores = self.unpack_tensor_to_dict(score)
            scores = {f"score_{k}": scores[k] for k in scores.keys()}
            all_scores.append(scores)

            features = map(self.unpack_tensor_to_dict, features)
            features = reduce(self.concat_dicts, features)
            features = {f"features_{k}": features[k] for k in features.keys()}
            all_features.append(features)

        all_scores = reduce(self.sum_dicts, all_scores)
        all_features = reduce(self.concat_dicts, all_features)

        inputs.update(all_scores)
        inputs.update(all_features)

        return inputs
    
class OobleckDiscriminator(nn.Module):

    def __init__(
            self,
            in_channels=1,
            ):
        super().__init__()

        multi_scale_discriminator = MultiScaleDiscriminator(
            in_channels=in_channels,
            n_scales=3,
        )

        multi_period_discriminator = MultiPeriodDiscriminator(
            in_channels=in_channels,
            periods=[2, 3, 5, 7, 11]
        )

        # multi_resolution_discriminator = MultiScaleSTFTDiscriminator(
        #     filters=32,
        #     in_channels = in_channels,
        #     out_channels = 1,
        #     n_ffts = [2048, 1024, 512, 256, 128],
        #     hop_lengths = [512, 256, 128, 64, 32],
        #     win_lengths = [2048, 1024, 512, 256, 128]
        # )

        self.multi_discriminator = MultiDiscriminator(
            [multi_scale_discriminator, multi_period_discriminator], #, multi_resolution_discriminator],
            ["reals", "fakes"]
        )

    def loss(self, reals, fakes):
        inputs = {
            "reals": reals,
            "fakes": fakes,
        }

        inputs = self.multi_discriminator(inputs)

        scores_real = inputs["score_reals"]
        scores_fake = inputs["score_fakes"]

        features_real = inputs["features_reals"]
        features_fake = inputs["features_fakes"]

        dis_loss, gen_loss = get_hinge_losses(scores_real, scores_fake)
         
        feature_matching_distance = torch.tensor(0.)

        for _, (scale_real, scale_fake) in enumerate(zip(features_real, features_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda real, fake: abs(real - fake).mean(),
                    scale_real,
                    scale_fake,
                )) / len(scale_real)
            
        return dis_loss, gen_loss, feature_matching_distance
    

## Discriminators from Descript Audio Codec repo
## Copied and modified under MIT license, see LICENSES/LICENSE_DESCRIPT.txt
class MPD(nn.Module):
    def __init__(self, period, channels=1):
        super().__init__()

        from dac.model.discriminator import WNConv2d

        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(channels, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MSD(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 44100, channels=1):
        super().__init__()

        from dac.model.discriminator import WNConv1d

        self.convs = nn.ModuleList(
            [
                WNConv1d(channels, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        x = AudioSignal(x, self.sample_rate)
        x.resample(self.sample_rate // self.rate)
        x = x.audio_data

        fmap = []

        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
        channels: int = 1
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        from dac.model.discriminator import WNConv2d
        from audiotools import STFTParams

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        self.channels = channels

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        from audiotools import AudioSignal
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())
        x = rearrange(x, "b ch f t c -> (b ch) c t f", ch=self.channels)
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap

# Adapted from https://github.com/open-mmlab/Amphion/blob/main/models/vocoders/gan/discriminator/mssbcqtd.py under the MIT license.
#   LICENSE is in incl_licenses directory.
class DiscriminatorCQT(nn.Module):
    def __init__(self, cfg: dict, hop_length: int, n_octaves:int, bins_per_octave: int):
        super().__init__()
        self.cfg = cfg

        self.filters = cfg["cqtd_filters"]
        self.max_filters = cfg["cqtd_max_filters"]
        self.filters_scale = cfg["cqtd_filters_scale"]
        self.kernel_size = (3, 9)
        self.dilations = cfg["cqtd_dilations"]
        self.stride = (1, 2)

        self.in_channels = cfg["cqtd_in_channels"]
        self.out_channels = cfg["cqtd_out_channels"]
        self.fs = cfg["sampling_rate"]
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.fmin = cfg["cqtd_fmin"]

        # Lazy-load
        from nnAudio import features

        self.cqt_transform = features.cqt.CQT2010v2(
            fmin=cfg["cqtd_fmin"],
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(nn.Conv2d(
                self.in_channels * 2,
                self.in_channels * 2,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            ))

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(
            self.in_channels * 2,
            self.filters,
            kernel_size=self.kernel_size,
            padding=self.get_2d_padding(self.kernel_size),
        ))

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(self.max_filters,
                (self.filters_scale ** (i + 1)) * self.filters)
            self.convs.append(weight_norm(nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=(dilation, 1),
                padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
            )))
            in_chs = out_chs

        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(weight_norm(nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
        )))

        self.conv_post = weight_norm(nn.Conv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
        ))

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)

        self.cqtd_normalize_volume = self.cfg.get("cqtd_normalize_volume", False)
        if self.cqtd_normalize_volume:
            print(f"[INFO] cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!")

    def get_2d_padding(self,
        kernel_size: Tuple[int, int],
        dilation: Tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)

        input_channels = x.shape[1]

        if input_channels >= 2:
            x = rearrange(x, 'b c ... -> (b c) ...')

        z = self.cqt_transform(x)
      
        if input_channels >= 2:
            z = rearrange(z, '(b c) ... -> b c ...', c = input_channels)

        z_amplitude = z[..., 0]
        z_phase = z[..., 1]

        if len(z_amplitude.shape) == 3:
            z_amplitude = z_amplitude.unsqueeze(1)
            z_phase = z_phase.unsqueeze(1)
        
        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            s = i * self.bins_per_octave
            e = (i + 1) * self.bins_per_octave
            latent_z.append(self.conv_pres[i](z[..., s:e]))
        latent_z = torch.cat(latent_z, dim=-1)

        fmap = []
        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)
            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)
        return latent_z, fmap

class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.cfg = cfg
        # Using get with defaults
        self.cfg["cqtd_filters"] = self.cfg.get("cqtd_filters", 32)
        self.cfg["cqtd_max_filters"] = self.cfg.get("cqtd_max_filters", 1024)
        self.cfg["cqtd_filters_scale"] = self.cfg.get("cqtd_filters_scale", 1)
        self.cfg["cqtd_dilations"] = self.cfg.get("cqtd_dilations", [1, 2, 4])
        self.cfg["cqtd_in_channels"] = self.cfg.get("cqtd_in_channels", 1)
        self.cfg["cqtd_out_channels"] = self.cfg.get("cqtd_out_channels", 1)
        # Multi-scale params to loop over
        self.cfg["cqtd_hop_lengths"] = self.cfg.get("cqtd_hop_lengths", [512, 256, 256])
        self.cfg["cqtd_n_octaves"] = self.cfg.get("cqtd_n_octaves", [9, 9, 9])
        self.cfg["cqtd_bins_per_octaves"] = self.cfg.get(
            "cqtd_bins_per_octaves", [24, 36, 48])
        self.cfg["cqtd_fmin"] = self.cfg.get("fmin", 32.7)

        n_discriminators = len(self.cfg["cqtd_hop_lengths"])
        self.discriminators = nn.ModuleList([DiscriminatorCQT(
            self.cfg,
            hop_length=self.cfg["cqtd_hop_lengths"][i],
            n_octaves=self.cfg["cqtd_n_octaves"][i],
            bins_per_octave=self.cfg["cqtd_bins_per_octaves"][i],
        ) for i in range(n_discriminators)])

    def forward(self, reals: torch.Tensor, gens: torch.Tensor, checkpointing: bool = False) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            if checkpointing:
                y_d_r, fmap_r = checkpoint(disc, reals)
                y_d_g, fmap_g = checkpoint(disc, gens)
            else:
                y_d_r, fmap_r = disc(reals)
                y_d_g, fmap_g = disc(gens)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def discriminator_loss(self, fake, real):
        y_real, y_fake, fmap_real, fmap_fake = self.forward(real, fake.clone().detach())

        loss_d = 0
        for x_fake, x_real in zip(y_fake, y_real):
            loss_d += torch.mean(x_fake ** 2)
            loss_d += torch.mean((1 - x_real) ** 2)
        loss_d /= len(y_fake)
        return loss_d

    def generator_loss(self, fake, real):
        y_real, y_fake, fmap_real, fmap_fake = self.forward(real, fake)

        loss_g = 0
        for x_fake in y_fake:
            loss_g += torch.mean((1 - x_fake) ** 2)

        counter = 0
        loss_feature = 0
        for i in range(len(fmap_fake)):
            for j in range(len(fmap_fake[i])):
                denominator = fmap_real[i][j].abs().mean().detach()
                loss_feature += F.l1_loss(fmap_fake[i][j], fmap_real[i][j].detach()) / denominator
                counter += 1
        loss_feature /= counter
        loss_g /= len(y_fake)
        return loss_g, loss_feature

    def loss(self, reals, fakes):
        gen_loss, feature_distance = self.generator_loss(fakes, reals)
        dis_loss = self.discriminator_loss(fakes, reals)
        return dis_loss, gen_loss, feature_distance

class DACDiscriminator(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 44100,
        bands: list = BANDS,
        **kwargs
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        discs += [MPD(p, channels=channels) for p in periods]
        discs += [MSD(r, sample_rate=sample_rate, channels=channels) for r in rates]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands, channels=channels) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [checkpoint(d,x) for d in self.discriminators]
        return fmaps

class DACGANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, use_hinge: bool = False, **discriminator_kwargs):
        super().__init__()
        self.use_hinge = use_hinge
        self.discriminator = DACDiscriminator(**discriminator_kwargs)

    def forward(self, fake, real):
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += (
                F.relu(x_fake[-1]).mean() +
                F.relu(1 - x_real[-1]).mean()
            ) if self.use_hinge else (
                (x_fake[-1] ** 2).mean() +
                ((1 - x_real[-1]) ** 2).mean()
            )
        loss_d /= len(d_fake)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += (
                F.relu(1 - x_fake[-1]).mean()
                if self.use_hinge else
                ((1 - x_fake[-1]) ** 2).mean()
            )

        n_discriminators = len(d_fake)
        loss_feature = 0
        for i in range(n_discriminators):
            # Average over N model layers (except for the last item, which is logits).
            n_layers = len(d_fake[i]) - 1
            loss_feature += sum(map(
                lambda j: F.l1_loss(d_fake[i][j], d_real[i][j].detach()),
                range(n_layers)
            )) / n_layers

        # Average over K discriminators.
        loss_feature = loss_feature / n_discriminators

        loss_g /= len(d_fake)
        return loss_g, loss_feature

    def loss(self, reals, fakes):
        gen_loss, feature_distance = self.generator_loss(fakes, reals)
        dis_loss = self.discriminator_loss(fakes, reals)
        return dis_loss, gen_loss, feature_distance

class BigVGANDiscriminator(nn.Module):
    def __init__(self, sample_rate: int,
        channels: int = 1,
        use_hinge: bool = False,
        periods: List[int] = [2, 3, 5, 7, 11],
        **cqt_kwargs,
    ):
        super().__init__()

        # Use MPD discriminator from DAC GAN, disable others.
        self.mpd = DACGANLoss(use_hinge=use_hinge, sample_rate=sample_rate,
            periods=periods, rates=[], fft_sizes=[], channels = channels)

        self.cqt = MultiScaleSubbandCQTDiscriminator({
            "cqtd_in_channels": channels,
            "sampling_rate": sample_rate, **cqt_kwargs,
        })

    def loss(self, reals, fakes):
        cqt_dis_loss, cqt_gen_loss, cqt_feature_distance = self.cqt.loss(reals, fakes)
        mpd_dis_loss, mpd_gen_loss, mpd_feature_distance = self.mpd.loss(reals, fakes)
        return (
            mpd_dis_loss + cqt_dis_loss,
            mpd_gen_loss + cqt_gen_loss,
            mpd_feature_distance + cqt_feature_distance)


class FilterBankDiscriminator(nn.Module):
    def __init__(
        self,
        period: int,
        taps: int = 0,
        beta: float = 0.0,
        cutoff_freq: float = 0.0,
        kernel_sizes: tp.List[int] = [5, 5, 5, 5, 5],
        strides: tp.List[int] = [3, 3, 3, 3, 3, 1],
        channels: tp.List[int] = [32, 128, 256, 512, 1024, 1024],
        norm: str = "weight_norm",
        in_channels: int = 1,
    ):
        super().__init__()
        self.period = period
        if period == 1:
            self.pqmf = nn.Identity()
        else:
            assert taps > 0 and beta > 0.0 and cutoff_freq > 0.0
            self.pqmf = PQMF(subbands=period, taps=taps, beta=beta, cutoff_freq=cutoff_freq)
        
        if norm == "weight_norm":
            norm_f = weight_norm
        elif norm == "spectral_norm":
            norm_f = spectral_norm
        else:
            raise ValueError(f"Unknown norm: {norm}")
        self.in_channels = in_channels
        c_in = in_channels
        
        self.convs = nn.ModuleList([])
        for (ch, s, k) in zip(channels, strides, kernel_sizes):
            kernel_size = (1, k)
            padding = (0, get_padding(k))
            
            conv = nn.Conv2d(c_in, ch, kernel_size, (1, s), padding=padding)
            # nn.init.kaiming_normal_(conv.weight, nonlinearity='relu', a=LRELU_SLOPE)
            # conv.bias.data.zero_()
            conv = norm_f(conv)
            
            self.convs.append(conv)
            c_in = ch
        
        conv = nn.Conv2d(c_in, 1, (1, 3), 1, padding=(0, 1))
        # nn.init.kaiming_normal_(conv.weight, nonlinearity='linear')
        # conv.bias.data.zero_()
        self.conv_post = norm_f(conv)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        # x: [Batch, 1, Time]
        fmap = []

        #x = self.pqmf(x).unsqueeze(1)   # [B, 1, Period, T//Period]
        x = fold_channels_into_batch(x)
        x = self.pqmf(x)
        x = unfold_channels_from_batch(x, self.in_channels)
        if self.period == 1:
            x = x.unsqueeze(2)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class ChromaDiscriminator(nn.Module):
    def __init__(
        self,
        n_chroma: int,
        sample_rate: int,
        kernel_sizes: tp.List[int] = [5, 5, 5, 5, 5],
        strides: tp.List[int] = [3, 3, 3, 3, 3, 1],
        channels: tp.List[int] = [32, 128, 256, 512, 1024, 1024],
        norm: str = "weight_norm",
        in_channels: int = 1,
        **kwargs
    ):
        super().__init__()
        
        self.chroma = ChromaSpectrogram(sample_rate = sample_rate, n_fft = 4096, n_chroma = n_chroma, normalized = True)
        if norm == "weight_norm":
            norm_f = weight_norm
        elif norm == "spectral_norm":
            norm_f = spectral_norm
        else:
            raise ValueError(f"Unknown norm: {norm}")
        self.in_channels = in_channels
        c_in = in_channels
        
        self.convs = nn.ModuleList([])
        for (ch, s, k) in zip(channels, strides, kernel_sizes):
            kernel_size = (1, k)
            padding = (0, get_padding(k))
            
            conv = nn.Conv2d(c_in, ch, kernel_size, (1, s), padding=padding)
            # nn.init.kaiming_normal_(conv.weight, nonlinearity='relu', a=LRELU_SLOPE)
            # conv.bias.data.zero_()
            conv = norm_f(conv)
            
            self.convs.append(conv)
            c_in = ch
        
        conv = nn.Conv2d(c_in, 1, (1, 3), 1, padding=(0, 1))
        # nn.init.kaiming_normal_(conv.weight, nonlinearity='linear')
        # conv.bias.data.zero_()
        self.conv_post = norm_f(conv)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        # x: [Batch, 1, Time]
        fmap = []

        #x = self.pqmf(x).unsqueeze(1)   # [B, 1, Period, T//Period]
        x = fold_channels_into_batch(x)
        x = self.chroma(x)
        x = unfold_channels_from_batch(x, self.in_channels)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiFilterBankDiscriminator(nn.Module):
    def __init__(
        self,
        periods: tp.List[int] = [1, 2, 3, 5, 7, 11],
        taps: int = 256,
        beta: float = 8.0,
        cutoff_freqs: tp.List[float] = [0, 0.253881, 0.170546, 0.103881, 0.075310, 0.049338],
        kernel_sizes: tp.List[int] = [5, 5, 5, 5, 5],
        strides: tp.List[int] = [3, 3, 3, 3, 1],
        channels: tp.List[int] = [32, 128, 512, 1024, 1024],
        norm: str = "weight_norm",
        in_channels: int = 1,
        **kwargs
    ):
        assert len(strides) == len(channels) == len(kernel_sizes)
        super().__init__()
        discs = [
            FilterBankDiscriminator(
                p, taps=taps, beta=beta, cutoff_freq=c, kernel_sizes=kernel_sizes,
                strides=strides, channels=channels, norm=norm, in_channels=in_channels
            ) for p, c in zip(periods, cutoff_freqs)
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y: torch.Tensor):
        y_ds = []
        fmaps = []
        for disc in self.discriminators:
            y_d, fmap = disc(y)
            y_ds.append(y_d)
            fmaps.append(fmap)

        return y_ds, fmaps


class HILDiscriminator(nn.Module):
    def __init__(self, normalize_losses = False, loss_type: tp.Literal["hinge", "rpgan", "sigmoid_rpgan"]="rpgan", add_noise = False, sample_rate = 44100, n_chroma = 0, *args, **kwargs):
        super().__init__()
        from .encodec import MultiScaleSTFTDiscriminator
        self.stft_discriminators = MultiScaleSTFTDiscriminator(*args, **kwargs)
        self.fb_discriminators = MultiFilterBankDiscriminator(*args, **kwargs)

        if add_noise:
            self.noise_model = PsychoacousticStereoNoise(sr = sample_rate)

        if n_chroma > 0:
            self.chroma_discriminator = ChromaDiscriminator(sample_rate = sample_rate, n_chroma = n_chroma, *args, **kwargs)

        self.normalize_losses = normalize_losses
        self.add_noise = add_noise
        self.loss_type = loss_type
        self.sample_rate = sample_rate

        if self.normalize_losses:
            self.fm_reduction = lambda x, y: abs(x - y).mean()/(abs(x).mean().detach() + 1e-3)
        else:
            self.fm_reduction = lambda x, y: abs(x - y).mean()

    def forward(self, x):
        if self.add_noise:
            with torch.no_grad():
                noise = self.noise_model(x)
            x = x + noise
        logits, features = self.stft_discriminators(x)
        logits_fb, features_fb = self.fb_discriminators(x)
        logits.extend(logits_fb)
        features.extend(features_fb)
        if hasattr(self, 'chroma_discriminator'):
            logits_chroma, features_chroma = self.chroma_discriminator(x)
            logits.append(logits_chroma)
            features.append(features_chroma)
        return logits, features

    def loss(self, reals, fakes):
        feature_matching_distance = torch.tensor(0., device=reals.device)
        logits_true, feature_true = self.forward(reals)
        logits_fake, feature_fake = self.forward(fakes)
        dis_loss = torch.tensor(0.,device=reals.device)
        adv_loss = torch.tensor(0.,device=reals.device)
        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):
            
            feature_matching_distance = feature_matching_distance + sum(
                map(
                    self.fm_reduction,
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            if self.loss_type == "hinge":

                _dis, _adv = get_hinge_losses(
                    logits_true[i],
                    logits_fake[i],
                ) 
            elif self.loss_type == "rpgan":

                _dis, _adv = get_relativistic_losses(
                    logits_true[i],
                    logits_fake[i],
                )
            elif self.loss_type == "sigmoid_rpgan":

                _dis, _adv = get_sigmoid_relgan_losses(
                    logits_true[i],
                    logits_fake[i],
                )
            else:
                raise ValueError(f"Unknown HILDiscriminator loss type: {loss_type}")

            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv
        feature_matching_distance = torch.nan_to_num(feature_matching_distance, nan=0.0, posinf=0.0)
        return dis_loss / len(logits_true), adv_loss / len(logits_true), feature_matching_distance / len(logits_true)


class MultiTransformerDiscriminator(nn.Module):
    def __init__(self, normalize_losses = False, loss_type: tp.Literal["hinge", "rpgan", "sigmoid_rpgan"]="rpgan", add_noise = False, noise_kwargs = {}, sample_rate = 44100, in_channels = 2,stft_kwargs = {}, patched_kwargs = {}, mfb_kwargs = {}, chroma_kwargs = {}, **kwargs):
        super().__init__()
        from .encodec import MultiScaleSTFTDiscriminator

        if stft_kwargs.get('enabled', False):
            self.stft_discriminators = TransformerMultiSTFTDiscriminator(in_channels = in_channels, **stft_kwargs)
        if patched_kwargs.get('enabled', True):
            self.patched_discriminators = TransformerMultiPatchedDiscriminator(in_channels = in_channels, **patched_kwargs)
        if mfb_kwargs.pop('enabled', False):
            if mfb_kwargs.pop('use_HIL', False):
                self.fb_discriminators = MultiFilterBankDiscriminator(in_channels = in_channels, **mfb_kwargs)
            else:
                self.fb_discriminators = TransformerMultiWaveletDiscriminator(in_channels = in_channels, **mfb_kwargs)
        if chroma_kwargs.get('enabled', False):
            self.chroma_discriminators = TransformerMultiChromaDiscriminator(in_channels = in_channels, **chroma_kwargs)

        self.normalize_losses = normalize_losses
        self.add_noise = add_noise
        if add_noise:
            self.noise_model = PsychoacousticStereoNoise(sr = sample_rate, **noise_kwargs)
        self.loss_type = loss_type
        self.sample_rate = sample_rate

        if self.normalize_losses:
            self.fm_reduction = lambda x, y: abs(x - y).mean()/(abs(x).detach().mean() + 1e-3)
        else:
            self.fm_reduction = lambda x, y: abs(x - y).mean()

        self.last_disc_loss = 1.0

    def forward(self, x):
        if self.add_noise:
            with torch.no_grad():
                noise = self.noise_model(x)
            x = x + noise
        if self.last_disc_loss < 0.1:
            x = x + torch.randn_like(x) * x.std() * 10 * (0.1 - self.last_disc_loss)
        logits, features = [], []
        if hasattr(self, 'stft_discriminators'):
            logits_stft, features_stft = self.stft_discriminators(x)
            logits.extend(logits_stft)
            features.extend(features_stft)
        if hasattr(self, 'patched_discriminators'):
            logits_patched, features_patched = self.patched_discriminators(x)
            logits.extend(logits_patched)
            features.extend(features_patched)
        if hasattr(self, 'fb_discriminators'):
            logits_fb, features_fb = self.fb_discriminators(x)
            logits.extend(logits_fb)
            features.extend(features_fb)
        if hasattr(self, 'chroma_discriminators'):
            logits_chroma, features_chroma = self.chroma_discriminators(x)
            logits.extend(logits_chroma)
            features.extend(features_chroma)
        return logits, features

    def loss(self, reals, fakes):
        feature_matching_distance = torch.tensor(0., device=reals.device)
        logits_true, feature_true = self.forward(reals)
        logits_fake, feature_fake = self.forward(fakes)
        dis_loss = torch.tensor(0.,device=reals.device)
        adv_loss = torch.tensor(0.,device=reals.device)
        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):
            feature_matching_distance = feature_matching_distance + sum(
                map(
                    self.fm_reduction,
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            if self.loss_type == "hinge":

                _dis, _adv = get_hinge_losses(
                    logits_true[i],
                    logits_fake[i],
                ) 
            elif self.loss_type == "rpgan":

                _dis, _adv = get_relativistic_losses(
                    logits_true[i],
                    logits_fake[i],
                )
            elif self.loss_type == "sigmoid_rpgan":

                _dis, _adv = get_sigmoid_relgan_losses(
                    logits_true[i],
                    logits_fake[i],
                )
            else:
                raise ValueError(f"Unknown HILDiscriminator loss type: {loss_type}")

            dis_loss = dis_loss + torch.nan_to_num(_dis, nan=0.0, posinf=0.0)
            adv_loss = adv_loss + torch.nan_to_num(_adv, nan=0.0, posinf=0.0)
        feature_matching_distance = torch.nan_to_num(feature_matching_distance, nan=0.0, posinf=0.0)
        self.last_disc_loss = 0.99 * self.last_disc_loss + 0.01 * dis_loss.item() / len(logits_true)
        return dis_loss / len(logits_true), adv_loss / len(logits_true), feature_matching_distance / len(logits_true)

class TransformerDiscriminator(nn.Module):
    def __init__(self, in_dim, transformer_dim, stride, pretransform, sliding_window = [1,1], depth = 3, checkpointing = False, differential = True,  max_depth_feature = 2, ff_mult = 1.5, dyt = False, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.transformer_dim = transformer_dim
        self.stride = stride
        self.sliding_window = sliding_window
        self.depth = depth
        self.checkpointing = checkpointing
        self.differential = differential
        self.discriminator = TransformerResamplingBlock(in_dim, transformer_dim, stride, sliding_window = sliding_window, transformer_depth = depth, checkpointing = checkpointing, differential = differential, use_flash = True, mask_noise = 0, ff_mult = ff_mult, dyt = dyt, dim_heads = 64, **kwargs)
        self.decision_map = torch.nn.Conv1d(transformer_dim, 1, 1,bias = False)
        self.pretransform = pretransform
        self.max_depth_feature = max_depth_feature

    def forward(self, x):
        if x.shape[-1] % self.pretransform.downsampling_ratio != 0:
            target_length = ((x.shape[-1] // self.pretransform.downsampling_ratio) + 1) * self.pretransform.downsampling_ratio
            x = F.pad(x, (0, target_length - x.shape[-1]), mode = 'constant')
        if hasattr(self, "pretransform") and self.pretransform is not None:
            x = self.pretransform(x)
        x, fmap = self.discriminator(x, return_features = True)
        logits = self.decision_map(x)
        return logits, fmap[:min(len(fmap),self.max_depth_feature)]


class TransformerMultiSTFTDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, n_ffts: tp.List[int] = [4096, 1024, 128], depths: tp.List[int] = [3, 3, 3], strides: tp.List[int] = [2, 32, 2], sliding_window_widths: tp.List[int] = [2, 4, 3], **kwargs):
        super().__init__()
        win_lengths = n_ffts

        self.discriminators = nn.ModuleList([])

        for i in range(len(n_ffts)):
            pretransform = ComplexSTFTPretransform(in_channels, n_ffts[i], ema_flatten = False, use_compander = False, demodulate = True, center = False)
            transformer_dim = get_transformer_dim(pretransform.encoded_channels)
            transformer = TransformerDiscriminator(pretransform.encoded_channels, transformer_dim = transformer_dim, stride = strides[i], pretransform = pretransform, depth = depths[i], sliding_window = [sliding_window_widths[i],sliding_window_widths[i]], **kwargs)
            self.discriminators.append(transformer)

        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor):
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps

class TransformerMultiPatchedDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, patch_sizes: tp.List[int] = [29, 443, 953], depths: tp.List[int] = [3, 3, 3], strides: tp.List[int] = [2, 16, 2], sliding_window_widths: tp.List[int] = [4, 8, 1], **kwargs):
        super().__init__()

        self.discriminators = nn.ModuleList([])

        for i in range(len(patch_sizes)):
            pretransform = PatchedPretransform(in_channels, patch_sizes[i])
            transformer_dim = get_transformer_dim(pretransform.encoded_channels)
            transformer = TransformerDiscriminator(pretransform.encoded_channels, transformer_dim = transformer_dim, stride = strides[i], pretransform = pretransform, depth = depths[i], sliding_window = [sliding_window_widths[i],sliding_window_widths[i]], **kwargs)
            self.discriminators.append(transformer)
            
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor):
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps



class TransformerMultiWaveletDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, levels: tp.List[int] = [4, 8, 10], depths: tp.List[int] = [3, 3, 3], strides: tp.List[int] = [2, 32, 2], sliding_window_widths: tp.List[int] = [3, 8, 1], **kwargs):
        super().__init__()

        self.discriminators = nn.ModuleList([])

        for i in range(len(levels)):
            pretransform = WaveletPretransform(in_channels, levels[i])
            transformer_dim = get_transformer_dim(pretransform.encoded_channels)
            transformer = TransformerDiscriminator(pretransform.encoded_channels, transformer_dim = transformer_dim, stride = strides[i], pretransform = pretransform, depth = depths[i], sliding_window = [sliding_window_widths[i],sliding_window_widths[i]], **kwargs)
            self.discriminators.append(transformer)
            
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor):
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps
        

class ChromaPretransform(nn.Module):
    def __init__(self, in_channels: int, n_chroma: int = 64, sample_rate: int = 44100, n_fft: int = 4096, ctroct: float = 5.0, octwidth: float = 1.5, normalized: bool = True, norm: int = 1):
        super().__init__()
        self.chroma = ChromaSpectrogram(sample_rate = sample_rate, n_fft = n_fft, n_chroma = n_chroma, ctroct = ctroct, octwidth = octwidth, normalized = normalized, norm = norm)
        self.in_channels = in_channels
        self.encoded_channels = n_chroma * in_channels
        self.downsampling_ratio = n_fft // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fold_channels_into_batch(x)
        x = self.chroma(x)
        x = unfold_channels_from_batch(x, self.in_channels)
        x = rearrange(x, 'b c f t -> b (c f) t')
        return x



class TransformerMultiChromaDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, centres: tp.List[int] = [1.0, 5.0, 9.0], octwidths: tp.List[int] = [1.0, 1.5, 1.0], depths: tp.List[int] = [3, 3, 3], strides: tp.List[int] = [16, 8, 2], sliding_window_widths: tp.List[int] = [3, 3, 3], **kwargs):
        super().__init__()

        self.discriminators = nn.ModuleList([])

        for i in range(len(centres)):
            #pretransform = nn.Sequential(FoldChannelsIntoBatch(), ChromaSpectrogram(sample_rate = 44100, n_fft = 4096, n_chroma = 64, ctroct = centres[i], octwidth = octwidths[i], normalized = True, norm = 1), UnfoldChannelsFromBatch(in_channels), FoldChannelsIntoEmbedding())
            pretransform = ChromaPretransform(in_channels, n_chroma = 64, sample_rate = 44100, n_fft = 4096, ctroct = centres[i], octwidth = octwidths[i], normalized = True, norm = 1)
            encoded_channels = pretransform.encoded_channels
            transformer_dim = get_transformer_dim(encoded_channels)
            transformer = TransformerDiscriminator(encoded_channels, transformer_dim = transformer_dim, stride = strides[i], pretransform = pretransform, depth = depths[i], sliding_window = [sliding_window_widths[i],sliding_window_widths[i]], **kwargs)
            self.discriminators.append(transformer)
            
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor):
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps

def get_transformer_dim(
    c: int,
    *,
    k: float = 16.0,
    min_d: int = 192,
    max_d: int = 512,
    head_dim: int = 64,
) -> int:
    """
    Return d_model as a clean multiple of head_dim.

    Uses a saturating sqrt-law: d_raw = k * sqrt(c), but performs rounding and
    clamping in units of head_dim to avoid float/rounding glitches.
    """
    assert head_dim > 0, "head_dim must be positive"

    # base suggestion from √-law
    d_raw = k * math.sqrt(max(1.0, float(c)))

    # round to nearest multiple in *integer head units*
    q = int(math.floor(d_raw / head_dim + 0.5))  # nearest integer

    # clamp in head units so result stays a multiple after clamping
    q_min = math.ceil(min_d / head_dim)
    q_max = math.floor(max_d / head_dim)
    q = max(q_min, min(q_max, q))

    return q * head_dim

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size - 1) * dilation // 2


def get_2d_padding(
    kernel_size: tp.Tuple[int, int],
    dilation: tp.Tuple[int, int] = (1, 1)
) -> tp.Tuple[int, int]:
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2
    )


class FoldChannelsIntoBatch(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fold_channels_into_batch(x)

class UnfoldChannelsFromBatch(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return unfold_channels_from_batch(x, self.channels)

class FoldChannelsIntoEmbedding(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b c t f -> b (c f) t')
        return x
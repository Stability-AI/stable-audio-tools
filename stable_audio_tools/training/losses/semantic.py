import typing as tp

import audiotools
import torch
import torchaudio

from einops import rearrange
from torch.nn import functional as F
from torch import nn

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

class HubertLoss(nn.Module):
    def __init__(self,
        feature_ids: tp.Optional[tp.List[int]] = None,
        weight: float = 1.0,
        model_name: str = "HUBERT_LARGE"
    ):
        super().__init__()

        self.weight = weight
        self.feature_ids = feature_ids
        self.model_name = model_name
        
        # Load model based on the specified model name
        if self.model_name == "WAVLM_LARGE":
            bundle = torchaudio.pipelines.WAVLM_LARGE
        elif self.model_name == "HUBERT_LARGE":
            bundle = torchaudio.pipelines.HUBERT_LARGE
        elif self.model_name == "WAV2VEC2_LARGE_LV60K":
            bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        self.model = bundle.get_model()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = fold_channels_into_batch(x)
        y = fold_channels_into_batch(y)

        conv_features = (
            self.feature_ids is not None and
            len(self.feature_ids) == 1 and
            self.feature_ids[0] == -1)

        # Extract features from conv layers only.
        if conv_features:
            if self.model.normalize_waveform:
                x = nn.functional.layer_norm(x, x.shape)
                y = nn.functional.layer_norm(y, y.shape)
            x_list, _ = self.model.model.feature_extractor(x, None)
            y_list, _ = self.model.model.feature_extractor(y, None)
            x_list = [x_list]
            y_list = [y_list]
        else:
            x_list, _ = self.model.extract_features(x)
            y_list, _ = self.model.extract_features(y)

        loss = 0
        denom = 0
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            if self.feature_ids is None or i in self.feature_ids or conv_features:
                loss += F.l1_loss(x, y) / (y.std() + 1e-5)
                denom += 1

        loss = loss / denom
        return self.weight * loss

# Implementation taken from:
# https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/loss.py#L231
class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    """

    def __init__(self, sample_rate: int,
        n_mels: tp.List[int],
        window_lengths: tp.List[int],
        loss_fn: tp.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        mel_fmin: tp.Optional[tp.List[float]] = None,
        mel_fmax: tp.Optional[tp.List[float]] = None,
        window_type: tp.Optional[str] = None,
    ):
        super().__init__()
        self.stft_params = [{
            "window_length": w,
            "hop_length": w // 4,
            "window_type": window_type,
        } for w in window_lengths]

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.pow = pow

        self.mel_fmin = (
            mel_fmin
            if mel_fmin is not None else
            [0.0 for _ in range(len(window_lengths))]
        )
        self.mel_fmax = (
            mel_fmax
            if mel_fmin is not None else
            [None for _ in range(len(window_lengths))]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = audiotools.AudioSignal(x, self.sample_rate)
        y = audiotools.AudioSignal(y, self.sample_rate)

        loss = 0.0
        for n_mels, fmin, fmax, params in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params,
        ):
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **params)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **params)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss

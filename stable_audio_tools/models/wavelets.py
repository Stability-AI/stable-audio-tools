"""The 1D discrete wavelet transform for PyTorch."""

from einops import rearrange
import pywt
import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal


def get_filter_bank(wavelet):
    filt = torch.tensor(pywt.Wavelet(wavelet).filter_bank)
    if wavelet.startswith("bior") and torch.all(filt[:, 0] == 0):
        filt = filt[:, 1:]
    return filt

class WaveletEncode1d(nn.Module):
    def __init__(self, 
                 channels, 
                 levels,
                 wavelet: Literal["bior2.2", "bior2.4", "bior2.6", "bior2.8", "bior4.4", "bior6.8"] = "bior4.4"):
        super().__init__()
        self.wavelet = wavelet
        self.channels = channels
        self.levels = levels
        filt = get_filter_bank(wavelet)
        assert filt.shape[-1] % 2 == 1
        kernel = filt[:2, None]
        kernel = torch.flip(kernel, dims=(-1,))
        index_i = torch.repeat_interleave(torch.arange(2), channels)
        index_j = torch.tile(torch.arange(channels), (2,))
        kernel_final = torch.zeros(channels * 2, channels, filt.shape[-1])
        kernel_final[index_i * channels + index_j, index_j] = kernel[index_i, 0]
        self.register_buffer("kernel", kernel_final)

    def forward(self, x):
        for i in range(self.levels):
            low, rest = x[:, : self.channels], x[:, self.channels :]
            pad = self.kernel.shape[-1] // 2
            low = F.pad(low, (pad, pad), "reflect")
            low = F.conv1d(low, self.kernel, stride=2)
            rest = rearrange(
                rest, "n (c c2) (l l2) -> n (c l2 c2) l", l2=2, c2=self.channels
            )
            x = torch.cat([low, rest], dim=1)
        return x


class WaveletDecode1d(nn.Module):
    def __init__(self, 
                 channels, 
                 levels,
                 wavelet: Literal["bior2.2", "bior2.4", "bior2.6", "bior2.8", "bior4.4", "bior6.8"] = "bior4.4"):
        super().__init__()
        self.wavelet = wavelet
        self.channels = channels
        self.levels = levels
        filt = get_filter_bank(wavelet)
        assert filt.shape[-1] % 2 == 1
        kernel = filt[2:, None]
        index_i = torch.repeat_interleave(torch.arange(2), channels)
        index_j = torch.tile(torch.arange(channels), (2,))
        kernel_final = torch.zeros(channels * 2, channels, filt.shape[-1])
        kernel_final[index_i * channels + index_j, index_j] = kernel[index_i, 0]
        self.register_buffer("kernel", kernel_final)

    def forward(self, x):
        for i in range(self.levels):
            low, rest = x[:, : self.channels * 2], x[:, self.channels * 2 :]
            pad = self.kernel.shape[-1] // 2 + 2
            low = rearrange(low, "n (l2 c) l -> n c (l l2)", l2=2)
            low = F.pad(low, (pad, pad), "reflect")
            low = rearrange(low, "n c (l l2) -> n (l2 c) l", l2=2)
            low = F.conv_transpose1d(
                low, self.kernel, stride=2, padding=self.kernel.shape[-1] // 2
            )
            low = low[..., pad - 1 : -pad]
            rest = rearrange(
                rest, "n (c l2 c2) l -> n (c c2) (l l2)", l2=2, c2=self.channels
            )
            x = torch.cat([low, rest], dim=1)
        return x
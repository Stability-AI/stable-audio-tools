import warnings
from typing import Optional
import math
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torchaudio
import torch.nn.functional as F
import torch.utils.data
from scipy.signal.windows import kaiser

#class STDCT(torch.jit.ScriptModule):
class STDCT(nn.Module):
    '''Short-Time Discrete Cosine Transform II
    forward(x, inverse=False):
        x: [B, 1, hop_size*T] or [B, hop_size*T]
        output: [B, N, T+1] (center = True)
        output: [B, N, T]   (center = False)
    forward(x, inverse=True):
        x: [B, N, T+1] (center = True)
        x: [B, N, T]   (center = False)
        output: [B, 1, hop_size*T]'''

    __constants__ = ["N", "hop_size", "padding"]

    def __init__(self, N: int, hop_size: int, win_size: Optional[int] = None,
                 win_type: Optional[str] = "hann", center: bool = False,
                 window: Optional[Tensor] = None, device=None, dtype=None):
        super().__init__()
        self.N = N
        self.hop_size = hop_size
        if center:
            self.padding = (N + 1) // 2     # <=> ceil{N / 2}
            self.output_padding = N % 2
        else:
            self.padding = (N - hop_size + 1) // 2  # <=> ceil{(N - hop_size) / 2}
            self.output_padding = (N - hop_size) % 2
            self.clip = (hop_size % 2 == 1)

        factory_kwargs = {'device': device, 'dtype': dtype}

        if win_size is None:
            win_size = N
        
        if window is not None:
            win_size = window.size(-1)
            if win_size < N:
                padding = N - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        elif win_type is None:
            window = torch.ones(N, dtype=torch.float32, device=device)
        else:
            window: Tensor = getattr(torch, f"{win_type}_window")(win_size, device=device)
            if win_size < N:
                padding = N - win_size
                window = F.pad(window, (padding//2, padding - padding//2))
        assert N >= win_size, f"N({N}) must be bigger than win_size({win_size})"
        n = torch.arange(N, dtype=torch.float32, device=device).view(1, 1, N)
        k = n.view(N, 1, 1)
        _filter = torch.cos(math.pi/N*k*(n+0.5)) * math.sqrt(2/N)
        _filter[0, 0, :] /= math.sqrt(2)
        dct_filter = (_filter * window.view(1, 1, N)).to(**factory_kwargs)
        window_square = window.square().view(1, -1, 1).to(**factory_kwargs)
        self.register_buffer('filter', dct_filter)
        self.register_buffer('window_square', window_square)
        self.filter: Tensor
        self.window_square: Tensor
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, hop_size*T] or [B, hop_size*T]
        # output: [B, N, T+1] (center = True)
        # output: [B, N, T]   (center = False)
        if x.dim() == 2:
            x = x.unsqueeze(1)
    
        x = F.conv1d(x, self.filter, bias=None, stride=self.hop_size,
            padding=self.padding)
        if self.clip:
            x = x[:, :, :-1]
        return x
    
    @torch.jit.export
    def inverse(self, spec: Tensor) -> Tensor:
        # x: [B, N, T+1] (center = True)
        # x: [B, N, T]   (center = False)
        # output: [B, 1, hop_size*T]
        wav =  F.conv_transpose1d(spec, self.filter, bias=None, stride=self.hop_size,
            padding=self.padding, output_padding=self.output_padding)
        B, T = spec.size(0), spec.size(-1)
        window_square = self.window_square.expand(B, -1, T)
        L = self.hop_size*T + (self.N-self.hop_size) - 2*self.padding + self.output_padding
        window_square_inverse = F.fold(
            window_square,
            output_size = (1, L),
            kernel_size = (1, self.N),
            stride = (1, self.hop_size),
            padding = (0, self.padding)
        ).squeeze(2)

        # NOLA(Nonzero Overlap-add) constraint
        assert torch.all(torch.ne(window_square_inverse, 0.0))
        return wav / window_square_inverse


class MDCT(torch.jit.ScriptModule):
    '''Modified Discrete Cosine Transform
    forward(x, inverse=False):
        y: [B, 1, N * T] -> pad N to left & right each.
        output: [B, N, T + 1]
    forward(x, inverse=True):
        y: [B, N, T + 1]
        output: [B, 1, N * T]'''

    __constants__ = ["N", "filter", "normalize"]

    def __init__(self, N: int, normalize: bool = True, device=None, dtype=None):
        super().__init__()
        self.N = N
        self.normalize = normalize

        k = torch.arange(N, dtype=torch.float32, device=device).view(N, 1, 1)
        n = torch.arange(2*N, dtype=torch.float32, device=device).view(1, 1, 2*N)
        mdct_filter = torch.cos(math.pi/N*(n+0.5+N/2)*(k+0.5))
        if normalize:
            mdct_filter /= math.sqrt(N)
        mdct_filter = mdct_filter.to(device=device, dtype=dtype)
        self.register_buffer("filter", mdct_filter)
        self.filter: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x, self.filter, bias=None, stride=self.N, padding=self.N)
    
    @torch.jit.export
    def inverse(self, x: Tensor) -> Tensor:
        if self.normalize:
            mdct_filter = self.filter
        else:
            mdct_filter = self.filter / self.N
        return F.conv_transpose1d(x, mdct_filter, bias=None, stride=self.N, padding=self.N)

def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    def __init__(self, subbands=4, taps=62, cutoff_freq=0.142, beta=9.0):
        super().__init__()
        h_proto = torch.from_numpy(design_prototype_filter(taps, cutoff_freq, beta)).to(dtype=torch.float32).unsqueeze(0)
        k = torch.arange(subbands, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(taps + 1, dtype=torch.float32).unsqueeze(0)
        pqmf_filter = 2 * h_proto * torch.cos(
            (2*k + 1) * np.pi / (2 * subbands) * (n - taps / 2) + (-1)**k * np.pi / 4
        ).unsqueeze(1) * subbands ** 0.5
        self.taps = taps
        self.subbands = subbands
        self.register_buffer("pqmf_filter", pqmf_filter)
        self.pqmf_filter: Tensor
    
    def forward(self, x: Tensor) -> Tensor:
        return self.analysis(x)
    
    def analysis(self, x: Tensor) -> Tensor:
        if x.dim() == 2:  # [B, T] -> [B, 1, T]
            x = x.unsqueeze(1)
        x = F.conv1d(x, self.pqmf_filter, None, stride=self.subbands, padding=self.taps//2)
        return x
    
    def synthesis(self, x: Tensor) -> Tensor:
        padding = self.taps // 2
        w = self.pqmf_filter
        x = F.conv_transpose1d(x, w, None, stride=self.subbands, padding=padding,
                               output_padding=self.subbands-1)
        return x

def dct1_rfft_impl(x):
    return torch.view_as_real(torch.fft.rfft(x, dim=1))

def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))

def idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def _sqrt_hann_tight(n_fft: int, hop_length: int, device=None, dtype=None) -> torch.Tensor:
    """g[n] = sqrt(Hann_periodic[n]) * sqrt(2/R),  R = n_fft / hop_length  ⇒  sum_m g^2[n-mH] == 1."""
    assert n_fft % hop_length == 0, "n_fft must be an integer multiple of hop_length (H=N/R)."
    R = n_fft // hop_length
    w = torch.hann_window(n_fft, periodic=True, device=device, dtype=dtype)  # 0.5 - 0.5 cos(2πn/N)
    g = w.clamp_min(0).sqrt() * math.sqrt(2.0 / R)
    return g

def _power_sine_tight(n_fft: int,  hop_length: int, device=None, dtype=None) -> torch.Tensor:
    """
    Tight (Parseval) power-sine analysis window for hop H=N/R (integer R>=2).

    g[n] = A_p * sin^p(pi n / N), with A_p chosen so that
    sum_{r=0}^{R-1} g^2[n - r*(N/R)] == 1 (interior samples).

    For R=4, the allowed powers are p ∈ {1,2,3}; p=1 recovers √Hann, p=3 gives
    lower sidelobes (slightly wider mainlobe).
    """
    R = n_fft // hop_length
    p = R - 1
    assert R >= 2 and 1 <= p <= (R - 1)
    N = int(n_fft)
    n = torch.arange(N, device=device, dtype=dtype)
    s = torch.sin(math.pi * n / N)  # matches periodic Hann endpoints
    # Mean of sin^{2p} over a period: C0 = (2p choose p)/2^{2p}
    from math import comb
    C0 = comb(2*p, p) / (2.0 ** (2*p))
    A = 1.0 / math.sqrt(R * C0)
    return (s ** p) * A

def _hermitian_sqrt(n_fft: int, device=None, dtype=None) -> torch.Tensor:
    """√2 for interior one-sided bins; 1 for DC (and Nyquist if even N)."""
    F1 = n_fft // 2 + 1
    s = torch.ones(F1, device=device, dtype=dtype)
    if n_fft % 2 == 0:
        if F1 > 2:
            s[1:-1] = math.sqrt(2.0)
    else:
        if F1 > 1:
            s[1:] = math.sqrt(2.0)
    return s

def tight_one_sided_complex_stft(x: torch.Tensor, n_fft: int, hop: int = None, center: bool = False, demod_mode: str = None) -> torch.Tensor:
    """
    Tight, one-sided, *demodulated* complex STFT for real signals, suitable for MRSTFT losses.

    Args
    ----
    x : [B, T] real tensor
    n_fft : FFT size (even). Hop is fixed to n_fft//2 with a power-sine tight window.
    center : passed to torch.stft
    demod_mode : 'sign' (default, exact parity demod for hop=N/2) or 'rot' (explicit complex rotation).

    Returns
    -------
    Xd : [B, F, M] complex tensor, demodulated per-bin to baseband and Hermitian-weighted for isometry.
    """
    assert n_fft % 2 == 0, "n_fft must be even"
    if hop is None:
        hop = n_fft // 2

    g = _power_sine_tight(n_fft, hop).to(x.device, x.dtype) 

    # Unitary, onesided STFT
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=g,
        center=center,
        pad_mode="reflect",
        normalized=True,
        onesided=True,
        return_complex=True,
    )  # [B, F, M]

    # Hermitian energy isometry for real signals (√2 on interior bins)
    herm = _hermitian_sqrt(n_fft).to(X.device, X.real.dtype)
    X = X * herm[:, None]  # broadcasts over time frames

    # Per-bin demodulation to baseband
    F, M = X.shape[-2], X.shape[-1]
    if demod_mode == "sign":
        # exact parity demod for hop=N/2: (-1)^{k·m}
        k_odd = (torch.arange(F, device=X.device, dtype=torch.int8) & 1).view(F, 1)
        m_odd = (torch.arange(M, device=X.device, dtype=torch.int8) & 1).view(1, M)
        parity = (k_odd & m_odd).to(X.real.dtype)
        sign = 1.0 - 2.0 * parity  # {+1, -1}
        X = X * sign
    elif demod_mode == "rot":
        # explicit complex rotation exp(-j 2π k m hop / n_fft)
        k = torch.arange(F, device=X.device, dtype=torch.float32).view(F, 1)
        m = torch.arange(M, device=X.device, dtype=torch.float32).view(1, M)
        phase = torch.remainder((2.0 * math.pi) * (k * (hop / float(n_fft))) * m, 2.0 * math.pi)
        rot = torch.complex(torch.cos(phase), -torch.sin(phase)).to(X.dtype)
        X = X * rot

    return X  # complex [B, F, M]

class TightSpectrogram(nn.Module):
    __constants__ = ["n_fft", "win_length", "hop_length", "pad", "normalized", "center", "pad_mode", "onesided"]

    def __init__(self,
                 n_fft: int = 1024,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn=None,
                 power: Optional[float] = None,
                 normalized: bool = True,
                 wkwargs=None,
                 center: bool = False,
                 pad_mode: str = "reflect",
                 onesided: bool = True,
                 demodulate: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = n_fft
        if win_length != n_fft and win_length is not None:
            warnings.warn("TightSpectrogram only supports win_length == n_fft; overriding win_length.")
        self.hop_length = n_fft // 2 
        if hop_length is not None and hop_length != self.hop_length:
            warnings.warn(f"TightSpectrogram only supports hop_length == n_fft//2 ({self.hop_length}); overriding hop_length.")
        self.pad = int(pad)
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.demodulate = demodulate

        # Register buffers (moved with .to())
        g = _power_sine_tight(self.win_length, self.hop_length)
        self.register_buffer("window", g, persistent = False)
        if self.onesided:
            herm = _hermitian_sqrt(self.n_fft)
            self.register_buffer("herm_sqrt", herm, persistent=False)
        else:
            self.register_buffer("herm_sqrt", torch.tensor(1.0), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [B, T] or [B, C, T] (real)
        returns:
            if power is None: complex STFT  [B, F, M] or [B, C, F, M]
            else: magnitude**power          same shapes (real)
        """
        x = waveform
        was_2d = (x.dim() == 2)
        if was_2d:
            x = x[:, None, :]  # [B, 1, T]

        B, C, T = x.shape
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad), mode = self.pad_mode)

        # Flatten channels for STFT call
        xc = x.reshape(B * C, -1)

        X = torch.stft(
            xc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(xc.device, xc.dtype),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )  # [B*C, F(=n_fft//2+1 if onesided else n_fft), M]

        # Hermitian energy isometry for one-sided (do nothing if two-sided)
        if self.onesided:
            X = X * self.herm_sqrt.to(X.device, X.dtype)[:, None]

        if self.demodulate:
            F, M = X.shape[-2], X.shape[-1]
            # exact parity demod for hop=N/2: (-1)^{k·m}
            k_odd = (torch.arange(F, device=X.device, dtype=torch.int8) & 1).view(F, 1)
            m_odd = (torch.arange(M, device=X.device, dtype=torch.int8) & 1).view(1, M)
            parity = (k_odd & m_odd).to(X.real.dtype)
            sign = 1.0 - 2.0 * parity  # {+1, -1}
            X = X * sign

        # Reshape back to [B, C, F, M]
        Freq, Frames = X.shape[-2], X.shape[-1]
        X = X.reshape(B, C, Freq, Frames)

        if self.power is None:
            out = X  # complex
        else:
            mag = X.abs()
            out = mag.pow(self.power)

        if was_2d:
            out = out[:, 0, ...]  # [B, F, M] (complex or real)
        return out


class ILDTransform(torch.nn.Module):
    def __init__(self, sample_rate=44100, n_fft=2048, n_mels=32, eps = 1e-3):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_fft // 2 + 1)
        self.eps = eps
        self.output_scale = abs(1.0 / math.log10(eps))

    def forward(self, x):
        x = self.mel_transform(x)
        x = torch.log10(x + self.eps)
        ild = (x[:,0,:] - x[:,1,:]) * self.output_scale
        return ild

class MeanChannelLog1pTransform(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = x.mean(dim=1)  # [B, F, T]
        return x.log1p()  # log(1 + x)
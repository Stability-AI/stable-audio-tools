# Copied and modified from https://github.com/csteinmetz1/auraloss/blob/main/auraloss/freq.py under Apache License 2.0
# You can find the license at LICENSES/LICENSE_AURALOSS.txt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Any
import scipy.signal
from .utils import mmd
from stable_audio_tools.models.transforms import tight_one_sided_complex_stft

def normalized_complex_distance_loss(x, y, eps=1e-5):
    numerator = (x-y).abs() ** 2
    return torch.log(numerator / numerator.std(dim = [-1,-2], keepdim=True).detach().clamp(min = eps) + 1).mean()


def if_gd_loss(Xp, Xr, eps=1e-3, w_floor=1e-3):
    # Xp, Xr: complex [..., F, T], float32
    # ---- time increments (IF) ----
    Rt_p = Xp[..., :, 1:] * torch.conj(Xp[..., :, :-1])
    Rt_r = Xr[..., :, 1:] * torch.conj(Xr[..., :, :-1])
    denom_t_p = (Xp[..., :, 1:].abs() * Xp[..., :, :-1].abs()).clamp_min(eps)
    denom_t_r = (Xr[..., :, 1:].abs() * Xr[..., :, :-1].abs()).clamp_min(eps)
    Ut_p = Rt_p / denom_t_p
    Ut_r = Rt_r / denom_t_r
    wt = torch.sqrt(denom_t_p * denom_t_r).clamp_min(w_floor).detach()
    wt = wt / wt.mean().clamp_min(1e-7)
    Lt = (1.0 - (Ut_p * torch.conj(Ut_r)).real) * wt

    # ---- frequency increments (GD) ----
    Rf_p = Xp[..., 1:, :] * torch.conj(Xp[..., :-1, :])
    Rf_r = Xr[..., 1:, :] * torch.conj(Xr[..., :-1, :])
    denom_f_p = (Xp[..., 1:, :].abs() * Xp[..., :-1, :].abs()).clamp_min(eps)
    denom_f_r = (Xr[..., 1:, :].abs() * Xr[..., :-1, :].abs()).clamp_min(eps)
    Uf_p = Rf_p / denom_f_p
    Uf_r = Rf_r / denom_f_r
    wf = torch.sqrt(denom_f_p * denom_f_r).clamp_min(w_floor).detach()
    wf = wf / wf.mean().clamp_min(1e-7)
    Lf = (1.0 - (Uf_p * torch.conj(Uf_r)).real) * wf

    return Lt.mean() + Lf.mean()


def apply_reduction(losses, reduction="none", retain_batch_dim=False):
    dim = [-1, -2] if retain_batch_dim and len(losses.shape) == 3 else None
    """Apply reduction to collection of losses."""
    if reduction == "mean":
        losses = losses.mean(dim = dim)
    elif reduction == "sum":
        losses = losses.sum(dim = dim)
    return losses

def get_window(win_type: str, win_length: int):
    """Return a window function.

    Args:
        win_type (str): Window type. Can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
        win_length (int): Window length

    Returns:
        win: The window as a 1D torch tensor
    """

    try:
        win = getattr(torch, win_type)(win_length)
    except:
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length))

    return win

class SumAndDifference(torch.nn.Module):
    """Sum and difference signal extraction module."""

    def __init__(self):
        """Initialize sum and difference extraction module."""
        super(SumAndDifference, self).__init__()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, #channels, #samples).
        Returns:
            Tensor: Sum signal.
            Tensor: Difference signal.
        """
        if not (x.size(1) == 2):  # inputs must be stereo
            raise ValueError(f"Input must be stereo: {x.size(1)} channel(s).")

        sum_sig = self.sum(x).unsqueeze(1)
        diff_sig = self.diff(x).unsqueeze(1)

        return sum_sig, diff_sig

    @staticmethod
    def sum(x):
        return x[:, 0, :] + x[:, 1, :]

    @staticmethod
    def diff(x):
        return x[:, 0, :] - x[:, 1, :]


class FIRFilter(nn.Module):
    """
    Psychoacoustic prefilter (hp/fd/A-weight/K-weight) as a fixed FIR, stable for AMP/DDP.
    - Fixes K-shelf gain (→ k, not k^2).
    - Uses firwin2 on a dense grid.
    - Reflect-padding to reduce edge transients.
    - Proper padding derived from kernel length.
    - Kernel registered as a buffer and cast to input dtype/device at runtime.
    """

    def __init__(self, filter_type="kw", coef=0.85, fs=44100, ntaps=257,
                 pad_mode: str = "reflect", ref_hz: float = 1000.0):
        super().__init__()
        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")
        self.filter_type = filter_type
        self.coef = float(coef)
        self.fs = float(fs)
        self.ntaps = int(ntaps)
        self.pad_mode = pad_mode
        self.ref_hz = float(ref_hz)

        # design FIR taps
        if filter_type == "hp":
            taps = np.zeros(2, dtype=np.float64)  # length-2 pre-emphasis [1, -a]
            taps[0] = 1.0
            taps[1] = -self.coef
        elif filter_type == "fd":
            # simple 2-sample difference y[n] = x[n] - a x[n-2]
            taps = np.zeros(3, dtype=np.float64)
            taps[0] = 1.0
            taps[2] = -self.coef
        elif filter_type in {"aw", "kw"}:
            taps = self._design_weighting_fir(filter_type)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        # normalise to unity gain at ref_hz (for hp/fd it's fine to skip if you prefer raw pre-emphasis)
        if filter_type in {"aw", "kw"}:
            w = 2 * np.pi * self.ref_hz / self.fs
            n = np.arange(len(taps))
            H_ref = np.abs(np.sum(taps * np.exp(-1j * w * n)))
            if H_ref > 0:
                taps = taps / H_ref

        # register as buffer, not parameter
        k = torch.from_numpy(taps.astype(np.float32))[None, None, :]
        self.register_buffer("kernel", k, persistent=False)

    # ---- design helpers ----

    def _design_weighting_fir(self, which: str) -> np.ndarray:
        fs = self.fs
        ntaps = self.ntaps

        if which == "aw":
            # Analog A-weight (IEC) as in many references
            f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
            A1000 = 1.9997  # dB
            NUMs = [(2*np.pi*f4)**2 * 10**(A1000/20), 0, 0, 0, 0]
            DENs = np.polymul([1, 4*np.pi*f4, (2*np.pi*f4)**2],
                              [1, 4*np.pi*f1, (2*np.pi*f1)**2])
            DENs = np.polymul(np.polymul(DENs, [1, 2*np.pi*f3]),
                              [1, 2*np.pi*f2])
        elif which == "kw":
            # Stage 1: 2nd-order HP (critical damping)
            f_hp, Q_hp = 38.135, 0.5
            w_hp = 2*np.pi*f_hp
            NUM_hp = [1, 0, 0]                  # s^2
            DEN_hp = [1, w_hp/Q_hp, w_hp**2]    # s^2 + (w/Q)s + w^2

            # Stage 2: high-shelf (→ gain k at HF, 1 at LF)  **FIXED: k, not k^2**
            f_shelf, Q_shelf, G_shelf = 1681.974, 1.69, 4.0
            k = 10**(G_shelf/20.0)
            w_s = 2*np.pi*f_shelf
            NUM_shelf = [k, (k*w_s)/Q_shelf, w_s**2]
            DEN_shelf = [1,    w_s /Q_shelf, w_s**2]

            NUMs = np.polymul(NUM_hp, NUM_shelf)
            DENs = np.polymul(DEN_hp, DEN_shelf)
        else:
            raise RuntimeError

        # Bilinear to digital IIR
        b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

        # --- Endpoint-safe grid for firwin2 ---
        freq = np.linspace(0.0, fs/2.0, num=8193, endpoint=True)  # Hz, exact 0 and fs/2
        _, H = scipy.signal.freqz(b, a, worN=freq, fs=fs)
        Hmag = np.abs(H)

        # FIR fit
        taps = scipy.signal.firwin2(ntaps, freq, Hmag, fs=fs)

        return taps

    # ---- forward ----

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (B, C, T) -> output: (B, C, T)
        """
        B, C, T = input.shape
        x = input.reshape(B*C, 1, T)

        # ensure kernel is on the right device/dtype (important for AMP)
        k = self.kernel.to(dtype=x.dtype, device=x.device)
        pad = (k.shape[-1] - 1) // 2

        if self.pad_mode in {"reflect", "replicate", "constant"}:
            mode = self.pad_mode if self.pad_mode != "constant" else "constant"
            x = F.pad(x, (pad, pad), mode=mode)
            y = F.conv1d(x, k, padding=0)
        else:
            y = F.conv1d(x, k, padding=pad)

        return y.reshape(B, C, -1)

    
class SpectralContrastLoss(torch.nn.Module):
    """Spectral contrast loss: symmetric, scale-invariant spectral distance.

    Computes the ratio ||x - y||_F / ||x + y||_F over the input magnitudes,
    where the numerator is the Frobenius norm of the difference and the
    denominator is the Frobenius norm of the sum.

    Args:
        eps (float, optional): Floor on the denominator norm to prevent NaN
            in near-silent regions. Default: 1e-4
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x_mag, y_mag):
        x_mag = x_mag.float()
        y_mag = y_mag.float()
        numerator = torch.norm(y_mag - x_mag, p="fro", dim=[-1, -2])
        denominator = torch.norm(x_mag + y_mag, p="fro", dim=[-1, -2]).clamp_min(self.eps)
        return (numerator / denominator).unsqueeze(-1).unsqueeze(-1)

# Backward-compatible alias
SpectralConvergenceLoss = SpectralContrastLoss

class STFTMagnitudeLoss(torch.nn.Module):
    """STFT magnitude loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)

    Log-magnitudes are calculated with `log(log_fac*x + log_eps)`, where `log_fac` controls the
    compression strength (larger value results in more compression), and `log_eps` can be used
    to control the range of the compressed output values (e.g., `log_eps>=1` ensures positive
    output values). The default values `log_fac=1` and `log_eps=0` correspond to plain log-compression.

    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        log_eps (float, optional): Constant value added to the magnitudes before evaluating the logarithm.
            Default: 0.0
        log_fac (float, optional): Constant multiplication factor for the magnitudes before evaluating the logarithm.
            Default: 1.0
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(self, log=True, log_eps=0.0, log_fac=1.0, distance="L1", reduction="mean"):
        super(STFTMagnitudeLoss, self).__init__()

        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac

        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        log_eps = x_mag.std(dim =[-1,-2], keepdim=True).detach().clamp(min=1e-4)**2 + y_mag.std(dim =[-1,-2], keepdim=True).detach().clamp(min=1e-4)**2
        log_eps = torch.sqrt(log_eps)
        if self.log:
            x_mag = torch.log(x_mag / log_eps + 1)
            y_mag = torch.log(y_mag / log_eps + 1)
        #if self.log:
        #    x_mag = torch.log(x_mag * self.log_fac + self.log_eps)
        #    y_mag = torch.log(y_mag * self.log_fac + self.log_eps)
        return self.distance(x_mag, y_mag)# + mmd(x_mag, y_mag, bandwidths = [0.001,0.01,0.1,1], dim = -1) #+ mmd(x_mag, y_mag, bandwidths = [0.001,0.01,0.1,1], dim = -2) 

class CepstralLoss(torch.nn.Module):
    def __init__(self, log_eps=0.0, log_fac=1.0, reduction="mean"):
        super(CepstralLoss, self).__init__()

        self.log_eps = log_eps
        self.log_fac = log_fac

        self.distance = torch.nn.L1Loss(reduction=reduction)

    def forward(self, x, y):
        x = torch.log(self.log_fac * x.abs() + self.log_eps)# * x / (1e-9 + x.abs())
        x_cep = torch.fft.irfft(x, dim = -2)#[:,:x_mag.size(-2),:]
        if y.std().item() >= 1e-6:
            y = torch.log(self.log_fac * y.abs() + self.log_eps)# * y / (1e-9 + y.abs())
            y_cep = torch.fft.irfft(y, dim = -2)#[:, :x_mag.size(-2), :]
        else:
            y_cep = torch.zeros_like(x_cep)
        dist = self.distance(x_cep, y_cep)# + mmd(x_cep, y_cep, bandwidths = [0.001,0.01,0.1,1], dim = -1)# + mmd(x_cep, y_cep, bandwidths = [0.001,0.01,0.1,1], dim = -2)  #/ 0.5* (x_cep.abs().mean(dim = -2).unsqueeze(1) + y_cep.abs().mean(dim = -2).unsqueeze(1))
        return dist 


class STFTLoss(torch.nn.Module):
    """STFT loss module.

    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).

    Args:
        fft_size (int, optional): FFT size in samples. Default: 1024
        hop_size (int, optional): Hop size of the FFT in samples. Default: 256
        win_length (int, optional): Length of the FFT analysis window. Default: 1024
        window (str, optional): Window to apply before FFT, can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of scaling frequency bins. Default: None.
        perceptual_weighting (bool, optional): Apply perceptual A-weighting (Sample rate must be supplied). Default: False
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            Default: 'mean'
        mag_distance (str, optional): Distance function ["L1", "L2"] for the magnitude loss terms.
        device (str, optional): Place the filterbanks on specified device. Default: None

    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'. By default.
        loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: int = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        output: str = "loss",
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Any = None,
        retain_batch_dim: bool = False,
        w_cep: float = 0.0,
        residual_loss: bool = False,
        **kwargs
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = get_window(window, win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.w_cep = w_cep
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.perceptual_weighting = perceptual_weighting
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.mag_distance = mag_distance
        self.device = device
        self.retain_batch_dim = retain_batch_dim
        self.residual_loss = residual_loss

        self.phs_used = bool(self.w_phs)

        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(
            log=True,
            reduction=reduction if not self.retain_batch_dim else "none",
            distance=mag_distance,
            **kwargs
        )
        self.linstft = STFTMagnitudeLoss(
            log=False,
            reduction=reduction if not self.retain_batch_dim else "none",
            distance=mag_distance,
            **kwargs
        )
        self.ceptstft = CepstralLoss(
            log_eps=1e-6,
            log_fac=1.0,
            reduction=reduction if not self.retain_batch_dim else "none"
        )

        # setup mel filterbank
        if scale is not None:
            try:
                import librosa.filters
            except Exception as e:
                print(e)
                print("Try `pip install auraloss[all]`.")

            if self.scale == "mel":
                assert sample_rate != None  # Must set sample rate to use mel scale
                assert n_bins <= fft_size  # Must be more FFT bins than Mel bins
                fb = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
                fb = torch.tensor(fb).unsqueeze(0)

            elif self.scale == "chroma":
                assert sample_rate != None  # Must set sample rate to use chroma scale
                assert n_bins <= fft_size  # Must be more FFT bins than chroma bins
                fb = librosa.filters.chroma(
                    sr=sample_rate, n_fft=fft_size, n_chroma=n_bins
                )

            else:
                raise ValueError(
                    f"Invalid scale: {self.scale}. Must be 'mel' or 'chroma'."
                )

            self.register_buffer("fb", fb, persistent = False)

        if scale is not None and device is not None:
            self.fb = self.fb.to(self.device)  # move filterbank to device

        if self.perceptual_weighting:
            if sample_rate is None:
                raise ValueError(
                    f"`sample_rate` must be supplied when `perceptual_weighting = True`."
                )
            self.prefilter = FIRFilter(filter_type="kw", fs=sample_rate)

    def stft(self, x):
        """Perform STFT.
        Args:
            x (Tensor): Input signal tensor (B, T).

        Returns:
            Tensor: x_mag, x_phs
                Magnitude and phase spectra (B, fft_size // 2 + 1, frames).
        """
        #x_stft = torch.stft(
        #    x,
        #    self.fft_size,
        #    self.hop_size,
        #    self.win_length,
        #    self.window,
        #    return_complex=True,
        #)
        #
        #x_mag = torch.sqrt(
        #    torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=self.eps)
        #)
        ### re-add sqrt(mag) + x_stft / sqrt(mag)

        x_stft = tight_one_sided_complex_stft(x, self.fft_size, hop = self.hop_size, center = False)
        x_mag = torch.clamp(x_stft.abs(), min = self.eps)

        return x_mag, x_stft

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        bs, chs, seq_len = input.size()
        input, target = input.float(), target.float() ## Ensure float32 for stability + BF16 training compatibility

        if self.perceptual_weighting:  # apply optional A-weighting via FIR filter
            # since FIRFilter only support mono audio we will move channels to batch dim
            input = input.view(bs * chs, 1, -1)
            target = target.view(bs * chs, 1, -1)

            # now apply the filter to both
            self.prefilter.to(input.device)
            input = self.prefilter(input)
            target = self.prefilter(target)

            # now move the channels back
            input = input.view(bs, chs, -1)
            target = target.view(bs, chs, -1)

        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(input.device)

        if self.residual_loss:
            input = target - input
            x_mag, x_phs = self.stft(input.view(-1, input.size(-1)))
            y_mag = torch.zeros_like(x_mag)
            y_phs = torch.zeros_like(x_phs)
        else:
            x_mag, x_phs = self.stft(input.view(-1, input.size(-1)))
            y_mag, y_phs = self.stft(target.view(-1, target.size(-1)))
        
        # apply relevant transforms
        if self.scale is not None:
            self.fb = self.fb.to(input.device)
            x_mag = torch.matmul(self.fb, x_mag)
            y_mag = torch.matmul(self.fb, y_mag)

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag**2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # compute loss terms
        sc_mag_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0
        phs_loss = (normalized_complex_distance_loss(x_phs,y_phs) + if_gd_loss(x_phs, y_phs)) if self.w_phs else 0.0
        cep_loss = self.ceptstft(x_phs, y_phs) if self.w_cep else 0.0

        # combine loss terms
        loss = (
            (self.w_sc * sc_mag_loss)
            + (self.w_log_mag * log_mag_loss)
            + (self.w_lin_mag * lin_mag_loss)
            + (self.w_phs * phs_loss)
            + (self.w_cep * cep_loss)
        )
        loss = apply_reduction(loss, reduction=self.reduction, retain_batch_dim=self.retain_batch_dim)

        
        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss

class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module.

    See [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str, optional): Window to apply before FFT, options include:
            'hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of mel frequency bins. Required when scale = 'mel'. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
    """

    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: List[int] = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        w_cep: float = 0.0, 
        **kwargs,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for i, (fs, ss, wl) in enumerate(zip(fft_sizes, hop_sizes, win_lengths)):
            self.stft_losses += [
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    window,
                    w_sc,
                    w_log_mag,
                    w_lin_mag,
                    w_phs,
                    sample_rate,
                    scale,
                    n_bins[i] if scale == "mel" and n_bins is not None else None,
                    perceptual_weighting,
                    scale_invariance,
                    w_cep = w_cep,
                    **kwargs,
                )
            ]

    def forward(self, x, y):
        mrstft_loss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []

        for f in self.stft_losses:
            if f.output == "full":  # extract just first term
                tmp_loss = f(x, y)
                mrstft_loss += tmp_loss[0]
                sc_mag_loss.append(tmp_loss[1])
                log_mag_loss.append(tmp_loss[2])
                lin_mag_loss.append(tmp_loss[3])
                phs_loss.append(tmp_loss[4])
            else:
                mrstft_loss += f(x, y)

        mrstft_loss /= len(self.stft_losses)

        if f.output == "loss":
            return mrstft_loss
        else:
            return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


class SumAndDifferenceSTFTLoss(torch.nn.Module):
    """Sum and difference sttereo STFT loss module.

    See [Steinmetz et al., 2020](https://arxiv.org/abs/2010.10291)

    Args:
        fft_sizes (List[int]): List of FFT sizes.
        hop_sizes (List[int]): List of hop sizes.
        win_lengths (List[int]): List of window lengths.
        window (str, optional): Window function type.
        w_sum (float, optional): Weight of the sum loss component. Default: 1.0
        w_diff (float, optional): Weight of the difference loss component. Default: 1.0
        perceptual_weighting (bool, optional): Apply perceptual A-weighting (Sample rate must be supplied). Default: False
        mel_stft (bool, optional): Use Multi-resoltuion mel spectrograms. Default: False
        n_mel_bins (int, optional): Number of mel bins to use when mel_stft = True. Default: 128
        sample_rate (float, optional): Audio sample rate. Default: None
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
    """

    def __init__(
        self,
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        window: str = "hann_window",
        w_sum: float = 1.0,
        w_diff: float = 1.0,
        output: str = "loss",
        **kwargs,
    ):
        super().__init__()
        self.sd = SumAndDifference()
        self.w_sum = w_sum
        self.w_diff = w_diff
        self.output = output
        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes,
            hop_sizes,
            win_lengths,
            window,
            **kwargs,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """This loss function assumes batched input of stereo audio in the time domain.

        Args:
            input (torch.Tensor): Input tensor with shape (batch size, 2, seq_len).
            target (torch.Tensor): Target tensor with shape (batch size, 2, seq_len).

        Returns:
            loss (torch.Tensor): Aggreate loss term. Only returned if output='loss'.
            loss (torch.Tensor), sum_loss (torch.Tensor), diff_loss (torch.Tensor):
                Aggregate and intermediate loss terms. Only returned if output='full'.
        """
        assert input.shape == target.shape  # must have same shape
        bs, chs, seq_len = input.size()

        # compute sum and difference signals for both
        input_sum, input_diff = self.sd(input)
        target_sum, target_diff = self.sd(target)

        # compute error in STFT domain
        sum_loss = self.mrstft(input_sum, target_sum)
        diff_loss = self.mrstft(input_diff, target_diff)
        loss = ((self.w_sum * sum_loss) + (self.w_diff * diff_loss)) / 2

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sum_loss, diff_loss


class SISDRLoss(torch.nn.Module):
    """Scale-invariant signal-to-distortion ratio loss module.

    Note that this returns the negative of the SI-SDR loss.

    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)

    Args:
        zero_mean (bool, optional) Remove any DC offset in the inputs. Default: ``True``
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, zero_mean=True, eps=1e-8, reduction="mean"):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean

        alpha = (input * target).sum(-1) / (((target ** 2).sum(-1)) + self.eps)
        target = target * alpha.unsqueeze(-1)
        res = input - target

        losses = 10 * torch.log10(
            (target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps
        )
        losses = apply_reduction(losses, self.reduction)
        return -losses


class SDSDRLoss(torch.nn.Module):
    """Scale-dependent signal-to-distortion ratio loss module.

    Note that this returns the negative of the SD-SDR loss.

    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)

    Args:
        zero_mean (bool, optional) Remove any DC offset in the inputs. Default: ``True``
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, zero_mean=True, eps=1e-8, reduction="mean"):
        super(SDSDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean

        alpha = (input * target).sum(-1) / (((target ** 2).sum(-1)) + self.eps)
        scaled_target = target * alpha.unsqueeze(-1)
        res = input - target

        losses = 10 * torch.log10(
            (scaled_target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps
        )
        losses = apply_reduction(losses, self.reduction)
        return -losses

class MelSTFTLoss(STFTLoss):
    """Mel-scale STFT loss module."""

    def __init__(
        self,
        sample_rate,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        n_mels=128,
        **kwargs,
    ):
        super(MelSTFTLoss, self).__init__(
            fft_size,
            hop_size,
            win_length,
            window,
            w_sc,
            w_log_mag,
            w_lin_mag,
            w_phs,
            sample_rate,
            "mel",
            n_mels,
            **kwargs,
        )

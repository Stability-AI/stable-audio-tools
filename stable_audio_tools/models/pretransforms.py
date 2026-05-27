import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from torchaudio.transforms import Resample, MelSpectrogram
from torch.nn.utils import weight_norm
import scipy.signal

from .wavelets import WaveletEncode1d, WaveletDecode1d
from .blocks import ResidualUnit, WNConv1d

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(x, channels):
    if channels == 1:
        return x.unsqueeze(1)
    x = rearrange(x, '(b c) ... -> b c ...', c = channels)
    return x

class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()

        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError
    
    def tokenize(self, x):
        raise NotImplementedError
    
    def decode_tokens(self, tokens):
        raise NotImplementedError

class AutoencoderPretransform(Pretransform):
    def __init__(self, model, scale=1.0, model_half=False, iterate_batch=False, chunked=False, enable_grad = False):
        super().__init__(enable_grad=enable_grad, io_channels=model.io_channels, is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete)
        self.model = model
        if not enable_grad:
            self.model.requires_grad_(False).eval()
        self.scale=scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate
        
        self.model_half = model_half
        self.iterate_batch = iterate_batch

        self.encoded_channels = model.latent_dim

        self.chunked = chunked
        self.num_quantizers = model.bottleneck.num_quantizers if model.bottleneck is not None and model.bottleneck.is_discrete else None
        self.codebook_size = model.bottleneck.codebook_size if model.bottleneck is not None and model.bottleneck.is_discrete else None

        if self.model_half:
            self.model.half()
    
    def encode(self, x, **kwargs):
        if self.model_half:
            x = x.half()
            self.model.to(torch.float16)

        encoded = self.model.encode_audio(x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            encoded = encoded.float()

        return encoded / self.scale

    def decode(self, z, **kwargs):
        z = z * self.scale

        if self.model_half:
            z = z.half()
            self.model.to(torch.float16)

        decoded = self.model.decode_audio(z, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            decoded = decoded.float()

        return decoded
    
    def tokenize(self, x, **kwargs):
        assert self.model.is_discrete, "Cannot tokenize with a continuous model"

        _, info = self.model.encode(x, return_info = True, **kwargs)

        return info[self.model.bottleneck.tokens_id]
    
    def decode_tokens(self, tokens, **kwargs):
        assert self.model.is_discrete, "Cannot decode tokens with a continuous model"

        return self.model.decode_tokens(tokens, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _pack_complex_to_channels(U: torch.Tensor) -> torch.Tensor:
    """Pack complex U[B,C,F,M] → real z[B,2·C·F,M] as [Re, Im] along channels."""
    B, C, F, M = U.shape
    Z = torch.empty((B, 2 * C * F, M), dtype=U.real.dtype, device=U.device)
    Z[:, 0::2, :] = U.real.reshape(B, C * F, M)
    Z[:, 1::2, :] = U.imag.reshape(B, C * F, M)
    return Z


def _unpack_channels_to_complex(Z: torch.Tensor, C: int, F: int) -> torch.Tensor:
    """Inverse of _pack_complex_to_channels. Z[B,2·C·F,M] → U[B,C,F,M] (complex)."""
    B, CF2, M = Z.shape
    assert CF2 == 2 * C * F, f"Z has {CF2} channels but expected {2*C*F}"
    Zre = Z[:, 0::2, :]
    Zim = Z[:, 1::2, :]
    return torch.complex(Zre, Zim).reshape(B, C, F, M)


def _sine_window(N: int, device, dtype) -> torch.Tensor:
    n = torch.arange(N, device=device, dtype=dtype)
    return torch.sin(math.pi * (n + 0.5) / N)


def _onesided_tight_weights(n_fft: int, device, dtype) -> torch.Tensor:
    """Return [1,1,F,1] weights: interior bins √2, DC/Nyquist 1 (for one-sided energy)."""
    F = n_fft // 2 + 1
    w = torch.ones(F, dtype=dtype, device=device)
    if (n_fft % 2) == 0:
        if F > 2:
            w[1:F-1] = math.sqrt(2.0)
    else:
        if F > 1:
            w[1:] = math.sqrt(2.0)
    return w.view(1, 1, F, 1)

def _demod_sign(F: int, M: int, device, dtype, expand_bc: bool = True) -> torch.Tensor:
    """Parity demod for hop=N/2.
    Returns ±1 with shape [1,1,F,M] (if expand_bc) for unambiguous broadcast
    against X[...,F,M]. Numerically exact for hop=N/2.
    """
    k_odd = (torch.arange(F, device=device, dtype=torch.int8) & 1).view(F, 1)
    m_odd = (torch.arange(M, device=device, dtype=torch.int8) & 1).view(1, M)
    parity = (k_odd & m_odd).to(torch.float32)
    sign = (1.0 - 2.0 * parity).to(dtype) # {0,1} -> {+1,-1}
    return sign.view(1, 1, F, M) if expand_bc else sign

def _to_mid_side(x: torch.Tensor) -> torch.Tensor:
    """x[B,2,T] → [B,2,T] with orthonormal mid/side."""
    s2 = math.sqrt(0.5)
    L, R = x[:, 0:1, :], x[:, 1:2, :]
    M = (L + R) * s2
    S = (L - R) * s2
    return torch.cat([M, S], dim=1)


def _from_mid_side(x: torch.Tensor) -> torch.Tensor:
    s2 = math.sqrt(0.5)
    M, S = x[:, 0:1, :], x[:, 1:2, :]
    L = (M + S) * s2
    R = (M - S) * s2
    return torch.cat([L, R], dim=1)


# -----------------------------------------------------------------------------
# Pretransform
# -----------------------------------------------------------------------------
class ComplexSTFTPretransform(Pretransform):  
    def __init__(
        self,
        channels: int,
        n_fft: int = 1024,
        demodulate: bool = True,
        center: bool = False,
        value_norm: str = "tight",   # 'tight' | 'none'
        use_mid_side: bool = False,
        ema_flatten: bool = True,
        flatten_alpha: float = 0.5,
        ema_beta: float = 1e-4,
        w_min: float = 0.25,
        w_max: float = 4.0,
        use_compander: bool = True,
        comp_alpha: float = 0.9,
        beta_min: float = 0.25,
        beta_max: float = 4.0,
        eps: float = 1e-12,
        enable_grad: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        gl_correction_steps: int = 0,  # NEW
        freeze_stats: bool = False,  # NEW
    ):
        super().__init__(enable_grad=enable_grad, io_channels=channels, is_discrete=False)
        self.C = int(channels)
        self.n_fft = int(n_fft)
        self.win_length = int(n_fft)
        self.hop_length = self.win_length // 2
        self.demodulate = bool(demodulate)
        self.center = bool(center)
        self.value_norm = value_norm
        self.use_mid_side = bool(use_mid_side)
        self.ema_flatten = bool(ema_flatten)
        self.flatten_alpha = float(flatten_alpha)
        self.ema_beta = float(ema_beta)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.use_compander = bool(use_compander)
        self.comp_alpha = float(comp_alpha)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.eps = float(eps)
        self.gl_correction_steps = int(max(0, gl_correction_steps))  # NEW

        _device = device if device is not None else torch.device("cpu")
        _dtype = dtype if dtype is not None else torch.float32

        # Fixed tight configuration: sine window, hop = n_fft//2
        win = _sine_window(self.win_length, _device, _dtype)
        self.register_buffer("window", win, persistent=False)

        if value_norm not in ("tight", "none"):
            raise ValueError("value_norm must be 'tight' or 'none'")

        # Bin weights for one-sided tightness
        w_one = _onesided_tight_weights(self.n_fft, _device, _dtype)
        self.register_buffer("_onesided_w", w_one, persistent=False)  # [1,1,F,1]

        # Derived sizes
        self.F = self.n_fft // 2 + 1
        self.downsampling_ratio = self.hop_length
        self.encoded_channels = self.C * (2 * self.F)

        # EMA stats buffers (shared across channels): shapes [1,1,F,1]
        self.register_buffer("psd", torch.ones(1,1,self.F,1, dtype=torch.float32, device=_device))        # E[|U|^2]
        self.register_buffer("s2a", torch.ones(1,1,self.F,1, dtype=torch.float32, device=_device))        # E[|Û|^{2α}] after flatten

        # Cache last decode length
        self._last_length: Optional[int] = None
        self.freeze_stats = bool(freeze_stats)

    # ---------------- STFT wrappers (unitary FFT) ----------------
    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = fold_channels_into_batch(x)  # [B·C,T]
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=True,       # unitary FFT
            return_complex=True,
        )  # [B,F,M]
        X = unfold_channels_from_batch(X, C)
        return X  # [B,C,F,M]

    def _istft(self, X: torch.Tensor, length: Optional[int]) -> torch.Tensor:
        B, C, F, M = X.shape
        X = fold_channels_into_batch(X)
        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=True,       # unitary FFT
            length=length,
            return_complex=False,
        )  # [B,T]
        x = unfold_channels_from_batch(x, C)
        return x  # [B,C,T]

    # ---------------- EMA helpers ----------------
    def _compute_W(self) -> Optional[torch.Tensor]:
        if not self.ema_flatten or self.flatten_alpha == 0.0:
            return None
        W = (self.psd + self.eps) ** (-0.5 * self.flatten_alpha)
        return torch.clamp(W, self.w_min, self.w_max)

    def _compute_Beta(self) -> Optional[torch.Tensor]:
        if not self.use_compander:
            return None
        Beta = (self.s2a + self.eps) ** (-0.5)
        return torch.clamp(Beta, self.beta_min, self.beta_max)

    # ---------------- NEW: GL post-correction helper ----------------
    def _gl_k_steps_time(self, X0: torch.Tensor, length: int, steps: int) -> torch.Tensor:
        """
        Run `steps` iterations of fixed-magnitude Griffin–Lim in RAW STFT.
        Keeps the STFT frame grid identical to X0 on all intermediate projections.
        Returns the final time-domain signal of length `length`.

        X0: [B,C,F,M] complex STFT (raw, i.e., after demod inverse etc.)
        steps: int >= 1
        """
        assert steps >= 1
        B, C, F, M = X0.shape
        hop = self.hop_length

        # Choose ISTFT length that guarantees STFT(...).shape[-1] == M
        if self.center:
            T_grid = (M - 1) * hop
        else:
            T_grid = (M - 1) * hop + self.win_length

        V = X0.abs()   # target magnitudes [B,C,F,M]
        X = X0         # current complex STFT on the same grid [B,C,F,M]

        for _ in range(steps):
            # Unit phasor (robust at zeros)
            ph = X / X.abs().clamp_min(self.eps)
            Z = V * ph
            # Project Z to consistency on the SAME frame grid
            x_mid = self._istft(Z, length=T_grid)
            X = self._stft(x_mid)  # guaranteed [B,C,F,M] by construction

        # Final synthesis to requested output length
        x_hat = self._istft(X, length=length)
        return x_hat

    # ---------------- API ----------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bfloat16:
            x = x.float()  # ensure float for fft when using bf16
        assert x.dim() == 3 and x.size(1) == self.C
        self._last_length = x.size(-1)

        # Optional mid–side (time domain)
        if self.use_mid_side and self.C == 2:
            x = _to_mid_side(x)

        X = self._stft(x)  # [B,C,F,M]
        if self.demodulate:
            rot = _demod_sign(self.F, X.size(-1), X.device, X.real.dtype, expand_bc=True)
            U = X * rot  # demod (unitary)
        else:
            U = X

        if self.value_norm == "tight":
            U = U * self._onesided_w  # one-sided weighting

        # --- EMA flattening (linear) ---
        if self.ema_flatten:
            if self.training and not self.freeze_stats:
                # FP32 stats to avoid fp16 under/overflow in pow
                with torch.cuda.amp.autocast(enabled=False):
                    P_hat = U.detach().abs().float().pow(2).mean(dim=(0,1,3), keepdim=True)  # [1,1,F,1]
                self.psd = (1.0 - self.ema_beta) * self.psd + self.ema_beta * P_hat
            W = self._compute_W()
            if W is not None:
                U = U * W
        else:
            W = None

        # --- Compander (radial power-law) ---
        if self.use_compander:
            # Update stats on **flattened** coefficients (FP32)
            if self.training and not self.freeze_stats:
                with torch.cuda.amp.autocast(enabled=False):
                    S_hat = U.detach().abs().float().pow(2 * self.comp_alpha).mean(dim=(0,1,3), keepdim=True)
                self.s2a = (1.0 - self.ema_beta) * self.s2a + self.ema_beta * S_hat

            Beta = self._compute_Beta()  # stays FP32 via buffers

            r = U.abs()
            # r^alpha in FP32, then cast back to real dtype for multiply
            with torch.cuda.amp.autocast(enabled=False):
                r_alpha = r.float().pow(self.comp_alpha)
            r_alpha = r_alpha.to(r.dtype)

            p = torch.where(r > 0, U / r.clamp_min(self.eps), torch.zeros_like(U))

            U = (Beta * r_alpha) * p

        return _pack_complex_to_channels(U)  # [B,2·C·F,M]


    def decode(self, z: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        B, EncC, M = z.shape
        T_len = length if length is not None else self._last_length
        if T_len is None:
            raise ValueError("Decode length unknown. Pass length= or call encode() first.")

        if z.dtype == torch.bfloat16:
            z = z.float()  # ensure float for fft when using bf16

        U = _unpack_channels_to_complex(z, C=self.C, F=self.F)

        # Inverse compander
        if self.use_compander:
            Beta = self._compute_Beta()  # FP32 via buffers
            r_prime = U.abs()

            p = torch.where(r_prime > 0, U / r_prime.clamp_min(self.eps), torch.zeros_like(U))

            # (r'/β)^(1/α) in FP32 to avoid fp16 blow-ups
            with torch.cuda.amp.autocast(enabled=False):
                rrec = torch.clamp(r_prime.float() / Beta, min=0.0).pow(1.0 / self.comp_alpha)
            rrec = rrec.to(r_prime.dtype)

            U = rrec * p

        # Inverse flattening
        if self.ema_flatten:
            W = self._compute_W()
            if W is not None:
                U = U / W

        if self.value_norm == "tight":
            U = U / self._onesided_w

        if self.demodulate:
            rot = _demod_sign(self.F, M, U.device, U.real.dtype, expand_bc=True)
            X = U * rot  # RAW STFT (re-modulate)
        else:
            X = U

        # -------- Optional GL post-correction (fixed magnitudes, k steps) --------
        steps = self.gl_correction_steps
        if steps > 0:
            x_hat = self._gl_k_steps_time(X, length=T_len, steps=steps)
        else:
            x_hat = self._istft(X, length=T_len)

        # Inverse mid–side
        if self.use_mid_side and self.C == 2:
            x_hat = _from_mid_side(x_hat)
        return x_hat


class WaveletPretransform(Pretransform):
    def __init__(self, channels, levels, wavelet = "bior4.4", enable_grad = False, **kwargs):
        super().__init__(enable_grad=False, io_channels=channels, is_discrete=False)

        self.encoder = WaveletEncode1d(channels, levels, wavelet)
        self.decoder = WaveletDecode1d(channels, levels, wavelet)

        self.downsampling_ratio = 2 ** levels
        self.io_channels = channels
        self.encoded_channels = channels * self.downsampling_ratio

    def encode(self, x):
        x = self.encoder(x) 
        return x
    
    def decode(self, z):
        return self.decoder(z)

class PatchedPretransform(Pretransform):
    def __init__(self, channels, patch_size, oversampling = 1, postfilter_channels = 0, **kwargs):
        super().__init__(enable_grad=True, io_channels=channels, is_discrete=False)
        self.channels = channels
        self.patch_size = patch_size
        self.oversampling = oversampling

        self.downsampling_ratio = patch_size
        self.io_channels = channels
        self.encoded_channels = channels * patch_size

        if self.oversampling > 1:
            self.input_upsampler = Resample(1, self.oversampling)
            self.output_downsampler = Resample(self.oversampling, 1)

        if postfilter_channels > 0:
            self.postfilter = nn.Sequential(
            WNConv1d(in_channels=channels, out_channels=postfilter_channels, kernel_size=7, padding=3, bias=True),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=1, use_snake=True, bias = True),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=3, use_snake=True, bias = True),
            ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels,
                         dilation=9, use_snake=True, bias = True),
            WNConv1d(in_channels=postfilter_channels, out_channels=channels, kernel_size=7, padding=3, bias=False))

    def _pad(self, x):
        seq_len = x.shape[-1]
        pad_len = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)
        return x
        
    def encode(self, x):
        if self.oversampling > 1:
            x = self.input_upsampler(x)
        x = self._pad(x)
        x = rearrange(x, "b c (l h) -> b (c h) l", h=self.patch_size)
        return x
    def decode(self, x):
        x = rearrange(x, "b (c h) l -> b c (l h)", h=self.patch_size)
        if hasattr(self, 'postfilter'):
            x = self.postfilter(x)
        if self.oversampling > 1:
            x = self.output_downsampler(x)
        return x

class HILPQMFPretransform(Pretransform):
    def __init__(self, channels, subbands, taps, beta, cutoff_freq):
        # TODO: Fix PQMF to take in in-channels
        super().__init__(enable_grad=False, io_channels=channels, is_discrete=False)
        from .transforms import PQMF
        self.channels = channels
        self.encoded_channels = channels * subbands
        self.pqmf = PQMF(subbands = subbands, taps = taps, beta = beta, cutoff_freq = cutoff_freq)

    def encode(self, x):
        # x is (Batch x Channels x Time)
        x = fold_channels_into_batch(x)
        pqmf = self.pqmf.analysis(x)
        pqmf = unfold_channels_from_batch(pqmf, self.channels)
        pqmf = rearrange(pqmf, 'b c f t -> b (c f) t')
        return pqmf

    def decode(self, z):
        z = rearrange(z, "b (c f) t -> b c f t", c=self.channels)
        z = fold_channels_into_batch(z)
        x = self.pqmf.synthesis(z)
        x = unfold_channels_from_batch(x, self.channels)
        return x


class PQMFPretransform(Pretransform):
    def __init__(self, attenuation=100, num_bands=16, channels = 1):
        # TODO: Fix PQMF to take in in-channels
        super().__init__(enable_grad=True, io_channels=channels, is_discrete=False)
        from .pqmf import PQMF
        self.pqmf = PQMF(attenuation, num_bands)


    def encode(self, x):
        # x is (Batch x Channels x Time)
        x = self.pqmf.forward(x)
        # pqmf.forward returns (Batch x Channels x Bands x Time)
        # but Pretransform needs Batch x Channels x Time
        # so concatenate channels and bands into one axis
        return rearrange(x, "b c n t -> b (c n) t")

    def decode(self, x):
        # x is (Batch x (Channels Bands) x Time), convert back to (Batch x Channels x Bands x Time) 
        x = rearrange(x, "b (c n) t -> b c n t", n=self.pqmf.num_bands)
        # returns (Batch x Channels x Time) 
        return self.pqmf.inverse(x)
        
class PretrainedDACPretransform(Pretransform):
    def __init__(self, model_type="44khz", model_bitrate="8kbps", scale=1.0, quantize_on_decode: bool = True, chunked=True):
        super().__init__(enable_grad=False, io_channels=1, is_discrete=True)
        
        import dac
        
        model_path = dac.utils.download(model_type=model_type, model_bitrate=model_bitrate)
        
        self.model = dac.DAC.load(model_path)

        self.quantize_on_decode = quantize_on_decode

        if model_type == "44khz":
            self.downsampling_ratio = 512
        else:
            self.downsampling_ratio = 320

        self.io_channels = 1

        self.scale = scale

        self.chunked = chunked

        self.encoded_channels = self.model.latent_dim

        self.num_quantizers = self.model.n_codebooks

        self.codebook_size = self.model.codebook_size

    def encode(self, x):

        latents = self.model.encoder(x)

        if self.quantize_on_decode:
            output = latents
        else:
            z, _, _, _, _ = self.model.quantizer(latents, n_quantizers=self.model.n_codebooks)
            output = z
        
        if self.scale != 1.0:
            output = output / self.scale
        
        return output

    def decode(self, z):
        
        if self.scale != 1.0:
            z = z * self.scale

        if self.quantize_on_decode:
            z, _, _, _, _ = self.model.quantizer(z, n_quantizers=self.model.n_codebooks)

        return self.model.decode(z)

    def tokenize(self, x):
        return self.model.encode(x)[1]
    
    def decode_tokens(self, tokens):
        latents = self.model.quantizer.from_codes(tokens)
        return self.model.decode(latents)
    
class AudiocraftCompressionPretransform(Pretransform):
    def __init__(self, model_type="facebook/encodec_32khz", scale=1.0, quantize_on_decode: bool = True):
        super().__init__(enable_grad=False, io_channels=1, is_discrete=True)
        
        try:
            from audiocraft.models import CompressionModel
        except ImportError:
            raise ImportError("Audiocraft is not installed. Please install audiocraft to use Audiocraft models.")
               
        self.model = CompressionModel.get_pretrained(model_type)

        self.quantize_on_decode = quantize_on_decode

        self.downsampling_ratio = round(self.model.sample_rate / self.model.frame_rate)

        self.sample_rate = self.model.sample_rate

        self.io_channels = self.model.channels

        self.scale = scale

        #self.encoded_channels = self.model.latent_dim

        self.num_quantizers = self.model.num_codebooks

        self.codebook_size = self.model.cardinality

        self.model.to(torch.float16).eval().requires_grad_(False)

    def encode(self, x):

        assert False, "Audiocraft compression models do not support continuous encoding"

        # latents = self.model.encoder(x)

        # if self.quantize_on_decode:
        #     output = latents
        # else:
        #     z, _, _, _, _ = self.model.quantizer(latents, n_quantizers=self.model.n_codebooks)
        #     output = z
        
        # if self.scale != 1.0:
        #     output = output / self.scale
        
        # return output

    def decode(self, z):
        
        assert False, "Audiocraft compression models do not support continuous decoding"

        # if self.scale != 1.0:
        #     z = z * self.scale

        # if self.quantize_on_decode:
        #     z, _, _, _, _ = self.model.quantizer(z, n_quantizers=self.model.n_codebooks)

        # return self.model.decode(z)

    def tokenize(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            return self.model.encode(x.to(torch.float16))[0]
    
    def decode_tokens(self, tokens):
        with torch.amp.autocast("cuda", enabled=False):
            return self.model.decode(tokens)

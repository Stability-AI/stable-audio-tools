import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

def _hann(n, device, dtype):
    # Hann window is always real-valued
    if dtype in [torch.complex64, torch.complex128]:
        dtype = torch.float32
    elif dtype not in [torch.float16, torch.float32, torch.float64]:
        dtype = torch.float32
    return torch.hann_window(n, periodic=True, device=device, dtype=dtype)

def _hz_to_bark(f_hz: torch.Tensor) -> torch.Tensor:
    """Convert frequency in Hz to Bark scale using Zwicker & Fastl formula"""
    f = f_hz * 0.001
    return 13.0 * torch.atan(0.76 * f) + 3.5 * torch.atan((f / 7.5) ** 2)

def _ath_db_spl(f_hz: torch.Tensor) -> torch.Tensor:
    """Absolute Threshold of Hearing in dB SPL (ISO 226:2003 approximation)"""
    f = f_hz.clamp(min=20.0, max=20000.0) / 1000.0
    # Standard ATH curve in dB SPL
    ath = (3.64 * (f ** -0.8)
           - 6.5 * torch.exp(-0.6 * (f - 3.3) ** 2)
           + 1e-3 * (f ** 4))
    return ath

def _critical_band_rate(f_hz: torch.Tensor) -> torch.Tensor:
    """Compute critical bandwidth at given frequencies (Zwicker & Terhardt)"""
    return 25.0 + 75.0 * (1.0 + 1.4 * (f_hz / 1000.0) ** 2) ** 0.69

def _build_spreading_function(freqs: torch.Tensor, device, dtype) -> torch.Tensor:
    n_freqs = len(freqs)
    
    # Vectorized computation
    bark_freqs = _hz_to_bark(freqs)
    
    # Create meshgrid for all frequency pairs
    bark_i = bark_freqs.unsqueeze(0)  # [1, n_freqs]
    bark_j = bark_freqs.unsqueeze(1)  # [n_freqs, 1]
    bark_dist = bark_j - bark_i  # [n_freqs, n_freqs]
    
    # Vectorized spreading computation
    upward_slope = -24.0
    downward_slope = -27.0
    
    spreading_db = torch.where(
        bark_dist > 0,
        downward_slope * bark_dist,
        upward_slope * torch.abs(bark_dist)
    )
    
    # Mask out very low frequencies
    freq_mask = (freqs.unsqueeze(0) >= 20).float()
    spreading = freq_mask * (10.0 ** (spreading_db / 10.0))
    
    # Normalize
    row_sums = spreading.sum(dim=1, keepdim=True).clamp_min(1e-5)
    return spreading / row_sums

class PsychoacousticStereoNoise(nn.Module):
    """
    Psychoacoustic noise generator that works directly in frequency domain
    to avoid filterbank reconstruction artifacts
    """
    def __init__(self,
                 sr: int = 44100,
                 n_fft: int = 2048,
                 hop: int = 512,
                 win: Optional[int] = None,
                 # Tonality detection
                 alpha_tonality: float = 0.3,  # Temporal smoothing
                 # Masking parameters  
                 tonal_masking_offset_db: float = -14.0,
                 noise_masking_offset_db: float = -6.0,
                 # Temporal masking
                 post_mask_time_ms: float = 30.0,  # Reduced from 100ms
                 # Stereo
                 enable_bmld: bool = True,
                 bmld_max_db: float = 6.0,
                 # Output control
                 snr_below_threshold_db: float = 10.0,  # Increased from 6dB for safety
                 min_noise_floor_db: float = -96.0,
                 high_freq_boost_db: float = 3.0,  # Boost high frequencies
                 adaptive_temporal: bool = True):  # Adaptive temporal masking
        super().__init__()
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.win = win or n_fft
        
        self.alpha_tonality = alpha_tonality
        self.tonal_masking_offset_db = tonal_masking_offset_db
        self.noise_masking_offset_db = noise_masking_offset_db
        self.post_mask_time_ms = post_mask_time_ms
        self.adaptive_temporal = adaptive_temporal
        self.enable_bmld = enable_bmld
        self.bmld_max_db = bmld_max_db
        self.snr_below_threshold_db = snr_below_threshold_db
        self.min_noise_floor_db = min_noise_floor_db
        self.high_freq_boost_db = high_freq_boost_db
        
        # Buffers
        self.register_buffer('_freqs', None, persistent=False)
        self.register_buffer('_bark_scale', None, persistent=False)
        self.register_buffer('_spreading', None, persistent=False)
        self.register_buffer('_ath_linear', None, persistent=False)
        self.register_buffer('_high_freq_boost', None, persistent=False)
        
    def _init_buffers(self, device, dtype):
        """Initialize frequency-domain buffers"""
        if self._freqs is None or self._freqs.device != device:
            n_freqs = self.n_fft // 2 + 1
            
            # Frequency bins
            freqs = torch.linspace(0, self.sr/2, n_freqs, device=device, dtype=dtype)
            self._freqs = freqs
            
            # Bark scale for each frequency
            self._bark_scale = _hz_to_bark(freqs)
            
            # Spreading function matrix (frequency to frequency)
            self._spreading = _build_spreading_function(freqs, device, dtype)
            
            # ATH curve
            ath_db = _ath_db_spl(freqs.clamp_min(20.0))
            self._ath_linear = 10.0 ** (-ath_db / 10.0)
            
            # High frequency compensation
            # Gradually boost high frequencies to compensate for poor resolution
            freq_norm = freqs / (self.sr / 2)
            boost_curve = torch.where(
                freq_norm > 0.5,  # Above Nyquist/2
                1.0 + (freq_norm - 0.5) * 2.0 * (self.high_freq_boost_db / 20.0),
                torch.ones_like(freq_norm)
            )
            self._high_freq_boost = boost_curve
    
    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT"""
        window = _hann(self.win, x.device, x.dtype)
        return torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop,
            win_length=self.win, window=window,
            return_complex=True, normalized=False
        )
    
    def _istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        """Compute ISTFT"""
        window = _hann(self.win, X.device, X.dtype)
        return torch.istft(
            X, n_fft=self.n_fft, hop_length=self.hop,
            win_length=self.win, window=window,
            length=length, return_complex=False
        )
    
    def _compute_tonality_freq(self, X: torch.Tensor) -> torch.Tensor:
        B, F, T = X.shape
        mag = X.abs().clamp_min(1e-5)

        # Vectorized local spectral flatness using convolution
        kernel_size = 5
        padding = kernel_size // 2

        # Pad magnitude spectrum
        mag_padded = torch.nn.functional.pad(mag, (0, 0, padding, padding), mode='reflect')

        # Unfold to get local windows
        mag_unfold = mag_padded.unfold(1, kernel_size, 1)  # [B, F, T, kernel_size]

        # Compute spectral flatness for all frequencies at once
        log_mag = torch.log(mag_unfold + 1e-5)
        geometric_mean = torch.exp(log_mag.mean(dim=-1))
        arithmetic_mean = mag_unfold.mean(dim=-1)

        sfm = (geometric_mean / (arithmetic_mean + 1e-5)).clamp(0, 1)
        tonality = 1.0 - sfm

        # Temporal smoothing (if needed) - can also be vectorized
        if self.alpha_tonality > 0:
            # Use exponential moving average filter
            tonality = self._ema_filter(tonality, self.alpha_tonality)

        return tonality.clamp(0, 1)

    def _ema_filter(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Vectorized exponential moving average"""
        B, F, T = x.shape
        output = torch.zeros_like(x)
        output[..., 0] = x[..., 0]

        # Recursive filter using scan-like operation
        for t in range(1, T):
            output[..., t] = alpha * output[..., t-1] + (1 - alpha) * x[..., t]

        return output
    
    def _compute_masking_threshold(self, X: torch.Tensor, 
                                  signal_rms: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-domain masking threshold
        X: [B, F, T] complex spectrum  
        signal_rms: [B, 1, 1] signal RMS
        Returns: threshold [B, F, T] in linear scale
        """
        B, F, T = X.shape
        
        # Power spectrum
        power = (X.abs() ** 2).real.clamp_min(1e-5)
        
        # Tonality estimation
        tonality = self._compute_tonality_freq(X)
        
        # Masking offset based on tonality
        offset_db = (tonality * self.tonal_masking_offset_db + 
                    (1 - tonality) * self.noise_masking_offset_db)
        offset_linear = 10.0 ** (offset_db / 10.0)
        
        # Excitation pattern (power adjusted by masking offset)
        excitation = power * offset_linear
        
        # Apply spreading function (frequency to frequency masking)
        # This is the key difference - we work directly in frequency domain
        spread_masking = torch.einsum('ij,bjt->bit', self._spreading, excitation)
        
        # ATH floor (scaled by signal level)
        ath_scaled = self._ath_linear.unsqueeze(0).unsqueeze(-1) * (signal_rms ** 2) * 0.01
        
        # Combine spreading and ATH
        threshold = torch.maximum(spread_masking, ath_scaled)
        
        # Apply high-frequency boost to compensate for poor HF masking
        threshold = threshold * self._high_freq_boost.unsqueeze(0).unsqueeze(-1)
        
        return threshold

    def _apply_temporal_masking(self, threshold: torch.Tensor, 
                                X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Memory-efficient temporal masking using in-place operations
        """
        B, F, T = threshold.shape
        
        if T == 1:
            return threshold
        
        # Calculate decay
        decay_samples = self.post_mask_time_ms * 0.001 * self.sr / self.hop
        decay_base = math.exp(-1.0 / max(1, decay_samples))
        
        # Work in-place on a clone
        output = threshold.clone()
        
        if self.adaptive_temporal and X is not None:
            # Compute energy changes for transient detection
            energy = (X.abs() ** 2).mean(dim=1, keepdim=True)  # [B, 1, T]
            
            # Pre-compute decay values to avoid repeated computation
            decay_trans = math.exp(-1.0 / max(1, decay_samples * 0.3))
            
            # Process sequentially but with vectorized operations per frame
            # This is memory efficient and still faster than the original
            prev_frame = output[..., 0].clone()
            
            for t in range(1, T):
                # Check for transients at this frame
                if t == 1:
                    energy_diff = (energy[..., 1] - energy[..., 0]).abs()
                    is_trans = energy_diff > energy[..., 0] * 0.5
                else:
                    energy_diff = (energy[..., t] - energy[..., t-1]).abs() 
                    is_trans = energy_diff > energy[..., t-1] * 0.5
                
                # Compute decay for this frame (broadcasting efficiently)
                decay = torch.where(is_trans, decay_trans, decay_base)
                
                # Apply masking
                torch.maximum(output[..., t], prev_frame * decay, out=output[..., t])
                
                # Update prev_frame for next iteration
                prev_frame = output[..., t]
        else:
            # Simple non-adaptive case - minimal memory usage
            # Use torch.jit.script to speed up the loop if available
            for t in range(1, T):
                torch.maximum(
                    output[..., t], 
                    output[..., t-1] * decay_base,
                    out=output[..., t]
                )
        
        return output

    def _compute_bmld(self, XL: torch.Tensor, XR: torch.Tensor) -> torch.Tensor:
        """
        Compute Binaural Masking Level Difference
        Returns: unmasking factor [B, F, T]
        """
        if not self.enable_bmld:
            return torch.ones_like(XL.abs())
        
        # Interaural coherence
        power_L = (XL.abs() ** 2).clamp_min(1e-5)
        power_R = (XR.abs() ** 2).clamp_min(1e-5)
        cross = (XL * XR.conj()).abs()
        coherence = (cross / torch.sqrt(power_L * power_R)).clamp(0, 1)
        
        # BMLD is strongest at low frequencies for decorrelated signals
        freq_weight = torch.exp(-self._freqs.unsqueeze(0).unsqueeze(-1) / 1500.0)
        bmld_db = self.bmld_max_db * (1 - coherence) * freq_weight
        
        return 10.0 ** (-bmld_db / 10.0)
    
    def _generate_shaped_noise(self, threshold: torch.Tensor,
                              signal_rms: torch.Tensor) -> torch.Tensor:
        """
        Generate frequency-shaped noise
        """
        B, F, T = threshold.shape
        device = threshold.device
        dtype = threshold.real.dtype if threshold.is_complex() else threshold.dtype
        
        # Target noise level (below threshold)
        reduction = 10.0 ** (self.snr_below_threshold_db / 10.0)
        target_power = threshold / reduction
        
        # Minimum noise floor
        noise_floor = (signal_rms ** 2) * (10.0 ** (self.min_noise_floor_db / 10.0))
        target_power = torch.maximum(target_power, noise_floor)
        
        # Convert to amplitude
        target_amp = torch.sqrt(target_power.clamp_min(1e-5))
        
        # Random phase noise
        phase = torch.rand(B, F, T, device=device, dtype=dtype) * 2 * math.pi - math.pi
        noise = target_amp * (torch.cos(phase) + 1j * torch.sin(phase))
        
        # Zero DC and Nyquist
        noise[:, 0, :] = 0
        if F == self.n_fft // 2 + 1:
            noise[:, -1, :] = 0
        
        return noise
    
    @torch.no_grad()
    @autocast(device_type = 'cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate psychoacoustically shaped noise
        x: [B, C, T] input audio
        Returns: [B, C, T] shaped noise
        """
        B, C, T_samples = x.shape
        device = x.device
        dtype = x.dtype
        
        x = x.to(torch.float32)
        # Initialize
        self._init_buffers(device, torch.float32)
        
        # Ensure stereo
        if C == 1:
            x = x.repeat(1, 2, 1)
            C = 2
        
        # Signal RMS
        signal_rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-5)
        
        # STFT
        XL = self._stft(x[:, 0, :])
        XR = self._stft(x[:, 1, :])
        
        # Compute masking thresholds for each channel
        threshold_L = self._compute_masking_threshold(XL, signal_rms)
        threshold_R = self._compute_masking_threshold(XR, signal_rms)
        
        # Apply BMLD
        bmld_factor = self._compute_bmld(XL, XR)
        threshold_L = threshold_L * bmld_factor
        threshold_R = threshold_R * bmld_factor
        
        # Temporal masking (adaptive if enabled)
        threshold_L = self._apply_temporal_masking(threshold_L, XL if self.adaptive_temporal else None)
        threshold_R = self._apply_temporal_masking(threshold_R, XR if self.adaptive_temporal else None)
        
        # Generate shaped noise
        noise_L = self._generate_shaped_noise(threshold_L, signal_rms)
        noise_R = self._generate_shaped_noise(threshold_R, signal_rms)
        
        # ISTFT
        noise_l = self._istft(noise_L, T_samples)
        noise_r = self._istft(noise_R, T_samples)
        
        # Stack and match input channels
        noise = torch.stack([noise_l, noise_r], dim=1)[:, :C, :]
        
        return noise.to(dtype)

    @torch.no_grad()
    @autocast(device_type = 'cuda', enabled=False)
    def calc_threshold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate psychoacoustically shaped noise
        x: [B, C, T] input audio
        Returns: [B, C, T] shaped noise
        """
        B, C, T_samples = x.shape
        device = x.device
        dtype = x.dtype
        
        x = x.to(torch.float32)
        # Initialize
        self._init_buffers(device, torch.float32)
        
        # Ensure stereo
        if C == 1:
            x = x.repeat(1, 2, 1)
            C = 2
        
        # Signal RMS
        signal_rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-5)
        
        # STFT
        XL = self._stft(x[:, 0, :])
        XR = self._stft(x[:, 1, :])
        
        # Compute masking thresholds for each channel
        threshold_L = self._compute_masking_threshold(XL, signal_rms)
        threshold_R = self._compute_masking_threshold(XR, signal_rms)
        
        # Apply BMLD
        bmld_factor = self._compute_bmld(XL, XR)
        threshold_L = threshold_L * bmld_factor
        threshold_R = threshold_R * bmld_factor
        
        # Temporal masking (adaptive if enabled)
        threshold_L = self._apply_temporal_masking(threshold_L, XL if self.adaptive_temporal else None)
        threshold_R = self._apply_temporal_masking(threshold_R, XR if self.adaptive_temporal else None)
        
        # Stack and match input channels
        threshold = torch.stack([threshold_L, threshold_R], dim=1)[:, :C, :]
        
        return threshold.to(dtype)

# Test with visualization
def test_and_visualize():
    """Test and visualize the noise spectrum"""
    import matplotlib.pyplot as plt
    
    # Generate test signal
    sr = 44100
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))
    
    # Multi-tone test signal
    signal = torch.zeros(2, len(t))
    # Add tones at different frequencies
    for freq in [500, 1000, 2000, 4000, 8000]:
        signal[0] += 0.1 * torch.sin(2 * math.pi * freq * t)
        signal[1] += 0.1 * torch.sin(2 * math.pi * freq * t * 1.01)  # Slight detuning
    
    signal = signal.unsqueeze(0)  # [1, 2, T]
    
    # Generate noise
    generator = PsychoacousticStereoNoise(
        sr=sr,
        n_fft=2048,
        hop=512,
        snr_below_threshold_db=10.0,  # More conservative
        post_mask_time_ms=30.0,  # Shorter temporal masking
        high_freq_boost_db=3.0,
        adaptive_temporal=True,  # Adaptive for transients
        enable_bmld=True
    )
    
    with torch.no_grad():
        noise = generator(signal)
    
    # Compute spectra for visualization
    window = torch.hann_window(2048, periodic=True)
    
    # Signal spectrum
    X_signal = torch.stft(signal[0, 0, :], n_fft=2048, hop_length=512, 
                         win_length=2048, window=window, return_complex=True)
    # Noise spectrum  
    X_noise = torch.stft(noise[0, 0, :], n_fft=2048, hop_length=512,
                        win_length=2048, window=window, return_complex=True)
    
    # Power spectra in dB
    signal_power_db = 20 * torch.log10(X_signal.abs().mean(dim=-1) + 1e-5)
    noise_power_db = 20 * torch.log10(X_noise.abs().mean(dim=-1) + 1e-5)
    
    # Plot
    freqs = torch.linspace(0, sr/2, len(signal_power_db))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(freqs, signal_power_db, label='Signal', alpha=0.7)
    plt.plot(freqs, noise_power_db, label='Generated Noise', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Spectrum Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, sr/2])
    
    plt.subplot(1, 2, 2)
    plt.plot(freqs, noise_power_db)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Noise Power (dB)')
    plt.title('Generated Noise Spectrum')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, sr/2])
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    snr = 10 * torch.log10(signal.pow(2).mean() / (noise.pow(2).mean() + 1e-5))
    print(f"Overall SNR: {snr:.2f} dB")
    print(f"Noise RMS: {noise.std():.6f}")
    print(f"High-freq noise (>10kHz) RMS: {noise[..., -sr//4:].std():.6f}")
    
    return signal, noise, sr


if __name__ == "__main__":
    signal, noise, sr = test_and_visualize()
import torch
import math
import typing as tp
from tqdm import trange, tqdm
import torch.distributions as dist

from . import k_diffusion as K

from ..data.utils import create_padding_mask_from_lengths, compute_effective_seq_len_from_conditioning

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def latent_size_t_shift(self, t, latent_dim, ref_latent_dim = 64):
    alpha = math.sqrt(latent_dim/ref_latent_dim)
    t = 1 - t  # invert to match noise at t=1, data at t=0
    return 1.0 - alpha * t / (1 + (alpha - 1) * t) # map and then invert again


class IdentityDistributionShift:
    """No-op distribution shift — returns timesteps unchanged."""
    def shift(self, t: torch.Tensor, seq_len):
        return t


class FluxDistributionShift:
    """Flux/SD3/Self-Flow timestep shift: t_shifted = alpha * t / (1 + (alpha-1) * t).

    Convention: t=0 is data, t=1 is noise.
    alpha > 1 shifts timesteps toward noise, appropriate for longer sequences
    where the critical structure-from-noise transition happens at higher noise levels.

    Can be used in two ways:
    - Constant alpha: set alpha_min == alpha_max. This is how the Self-Flow paper
      (BFL, 2025) uses it, with alpha chosen per modality/autoencoder.
      Reference values from the paper: audio sampleshift=6.93, trainshift=1.0;
      video sampleshift=15.0, trainshift=2.95; images sampleshift=1.78-6.93.
    - Seq_len-dependent alpha: set different alpha_min/alpha_max. Alpha is
      interpolated log-linearly in seq_len space (power-law), following the
      SD3 derivation where alpha ∝ sqrt(seq_len).

    Args:
        min_length: Minimum sequence length (alpha = alpha_min here)
        max_length: Maximum sequence length (alpha = alpha_max here)
        alpha_min: Shift factor at min_length (1.0 = no shift)
        alpha_max: Shift factor at max_length (1.0 = no shift)
    """
    def __init__(self, min_length=256, max_length=4096,
                 alpha_min=1.0, alpha_max=1.0):
        self.min_length = min_length
        self.max_length = max_length
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        # Precompute for log-linear interpolation
        self.log_alpha_min = math.log(max(alpha_min, 1e-8))
        self.log_alpha_max = math.log(max(alpha_max, 1e-8))
        self.log_min_seq = math.log(min_length)
        self.log_max_seq = math.log(max_length)
        if self.log_max_seq == self.log_min_seq:
            self.log_max_seq += 1e-8  # prevent division by zero for constant alpha

    def get_alpha(self, seq_len: tp.Union[int, torch.Tensor]):
        """Compute alpha via log-linear interpolation in seq_len."""
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.float().clamp(self.min_length, self.max_length)
            log_seq = torch.log(seq_len)
            frac = (log_seq - self.log_min_seq) / (self.log_max_seq - self.log_min_seq)
            log_alpha = self.log_alpha_min + frac * (self.log_alpha_max - self.log_alpha_min)
            return torch.exp(log_alpha)
        else:
            seq_len = max(min(seq_len, self.max_length), self.min_length)
            log_seq = math.log(seq_len)
            frac = (log_seq - self.log_min_seq) / (self.log_max_seq - self.log_min_seq)
            log_alpha = self.log_alpha_min + frac * (self.log_alpha_max - self.log_alpha_min)
            return math.exp(log_alpha)

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        """Shift timesteps based on sequence length.

        Args:
            t: Timesteps tensor of shape (batch_size,) or (steps,)
            seq_len: Either a scalar int (same shift for all elements) or
                     tensor of shape (batch_size,) for per-element shifts
        Returns:
            Shifted timesteps. If seq_len is a tensor and t is 1D with different size,
            returns shape (batch_size, steps) for per-element schedules.
        """
        alpha = self.get_alpha(seq_len)

        if isinstance(seq_len, torch.Tensor):
            alpha = alpha.to(t.device)
            if t.dim() == 1 and alpha.dim() == 1 and t.shape[0] != alpha.shape[0]:
                t = t.unsqueeze(0)
                alpha = alpha.unsqueeze(1)

        return alpha * t / (1 + (alpha - 1.0) * t)


class DistributionShift:
    def __init__(self, base_shift=0.5, max_shift=1.15, max_length=4096, min_length=256, use_sine=False):
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.max_length = max_length
        self.min_length = min_length
        self.use_sine = use_sine

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        """
        Shift timesteps based on sequence length to adjust noise schedule.

        Args:
            t: Timesteps tensor of shape (batch_size,) or (steps,)
            seq_len: Either a scalar int (same shift for all elements) or
                     tensor of shape (batch_size,) for per-element shifts
        Returns:
            Shifted timesteps. If seq_len is a tensor and t is 1D with different size,
            returns shape (batch_size, steps) for per-element schedules.
        """
        if isinstance(seq_len, torch.Tensor):
            # Per-element sequence lengths
            # Ensure seq_len is on the same device as t
            seq_len = seq_len.to(t.device)
            seq_len_clamped = seq_len.float().clamp(self.min_length, self.max_length)
            # Handle broadcasting when t and seq_len have different sizes
            if t.dim() == 1 and seq_len_clamped.dim() == 1 and t.shape[0] != seq_len_clamped.shape[0]:
                # t: (steps,) -> (1, steps), seq_len: (batch,) -> (batch, 1)
                # Result: (batch, steps)
                t = t.unsqueeze(0)
                seq_len_clamped = seq_len_clamped.unsqueeze(1)
            sigma = 1.0
            mu = - (self.base_shift + (self.max_shift - self.base_shift) * (seq_len_clamped - self.min_length) / (self.max_length - self.min_length))
            t_out = 1 - torch.exp(mu) / (torch.exp(mu) + (1 / (1 - t) - 1) ** sigma)
            if self.use_sine:
                t_out = torch.sin(t_out * math.pi / 2)
        else:
            # Scalar path (original behavior)
            seq_len = min(max(seq_len, self.min_length), self.max_length)
            sigma = 1.0
            mu = - (self.base_shift + (self.max_shift - self.base_shift) * (seq_len - self.min_length) / (self.max_length - self.min_length))
            t_out = 1 - math.exp(mu) / (math.exp(mu) + (1 / (1 - t) - 1) ** sigma)

            if self.use_sine:
                t_out = torch.sin(t_out * math.pi / 2)

        return t_out


class LogSNRShift:
    """Adaptive log-SNR distribution shift.

    Maps t∈[0,1] to log-SNR-spaced values while preserving order (0→0, 1→1).
    Equivalent to applying: logsnr = linspace(logsnr_end, logsnr_start, N)
    then t = sigmoid(-logsnr), which spaces steps uniformly in log-SNR.

    logsnr_start (the high-t bound) scales with sequence length following
    the "-1 per doubling" rule:
        logsnr_start = anchor_logsnr - rate * log₂(seq_len / anchor_length)

    This captures the empirical finding that the critical log-SNR point
    (where structure emerges from noise) drops by ~rate for each doubling
    of sequence length. logsnr_end (the low-t bound) is fixed because
    low-t refinement is purely local.
    """

    def __init__(self, anchor_length=2000, anchor_logsnr=-6.2,
                 rate=1.0, logsnr_end=2.0):
        self.anchor_length = anchor_length
        self.anchor_logsnr = anchor_logsnr
        self.rate = rate
        self.logsnr_end = logsnr_end

    def get_logsnr_start(self, seq_len):
        """Compute adaptive logsnr_start: drops by `rate` per doubling of seq_len."""
        if isinstance(seq_len, torch.Tensor):
            log2_ratio = torch.log2(seq_len.float() / self.anchor_length)
            return self.anchor_logsnr - self.rate * log2_ratio
        else:
            log2_ratio = math.log2(seq_len / self.anchor_length)
            return self.anchor_logsnr - self.rate * log2_ratio

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        """Transform t∈[0,1] to log-SNR-spaced t with adaptive bounds.

        Maps through: logsnr = logsnr_end - t * (logsnr_end - logsnr_start)
                      t_out = sigmoid(-logsnr)

        Preserves order: 0→~0, 1→~1, with exact endpoint preservation.

        Args:
            t: Timesteps tensor of shape (batch_size,) or (steps,)
            seq_len: Either a scalar int or tensor of shape (batch_size,)
        Returns:
            Log-SNR-spaced timesteps in [0, 1].
        """
        t_original = t
        logsnr_start = self.get_logsnr_start(seq_len)

        if isinstance(seq_len, torch.Tensor):
            logsnr_start = logsnr_start.to(t.device)
            if t.dim() == 1 and logsnr_start.dim() == 1 and t.shape[0] != logsnr_start.shape[0]:
                t = t.unsqueeze(0)
                logsnr_start = logsnr_start.unsqueeze(1)

        # Map t through log-SNR space (monotonically: low t → high logsnr → low t_out)
        logsnr = self.logsnr_end - t * (self.logsnr_end - logsnr_start)
        t_out = torch.sigmoid(-logsnr)

        # Preserve exact endpoints
        t_out = torch.where(t_original <= 0, torch.zeros_like(t_out), t_out)
        t_out = torch.where(t_original >= 1, torch.ones_like(t_out), t_out)

        return t_out


def build_schedule(
    steps: int,
    sigma_max: float = 1.0,
    dist_shift = None,
    effective_seq_len: tp.Union[int, torch.Tensor, None] = None,
    fallback_seq_len: tp.Optional[int] = None,
    include_endpoint: bool = True,
    device: tp.Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build a timestep schedule for diffusion sampling.

    Returns a 1D tensor of shape (N,) where N = steps+1 (if include_endpoint)
    or steps (if not), OR a 2D tensor of shape (batch_size, N) when
    effective_seq_len is a tensor and dist_shift produces per-element schedules.

    Args:
        steps: Number of sampling steps.
        sigma_max: Starting noise level (1.0 for full generation, <1.0 for variations).
        dist_shift: Optional distribution shift object (FluxDistributionShift,
            DistributionShift, LogSNRShift, etc.). Applied to warp the linear schedule.
        effective_seq_len: Sequence length for dist_shift. Scalar int or
            tensor of shape (batch_size,) for per-element schedules.
        fallback_seq_len: Fallback when effective_seq_len is None (typically x.shape[-1]).
        include_endpoint: If True, schedule includes 0 as final value (RF samplers).
            If False, excludes 0 (v-diffusion DDIM).
        device: Device for the output tensor.
    """
    n_points = steps + 1 if include_endpoint else steps

    if include_endpoint:
        t = torch.linspace(sigma_max, 0, n_points, device=device)
    else:
        t = torch.linspace(sigma_max, 0, n_points + 1, device=device)[:-1]

    if dist_shift is not None:
        seq_len = effective_seq_len if effective_seq_len is not None else fallback_seq_len
        if isinstance(seq_len, torch.Tensor):
            # Clamp per-element sequence lengths to avoid zeros causing log/NaN issues
            seq_len = torch.clamp(seq_len, min=1)
        elif seq_len is not None:
            # Clamp scalar sequence length to at least 1
            seq_len = max(int(seq_len), 1)
        t = dist_shift.shift(t, seq_len)

        # Ensure the first timestep remains aligned with sigma_max after shifting.
        # This keeps the schedule consistent with the initialization in sample_diffusion(),
        # which mixes init_data using sigma_max.
        if isinstance(t, torch.Tensor):
            sigma_max_tensor = t.new_tensor(sigma_max)
            if t.ndim == 1:
                t[0] = sigma_max_tensor
            else:
                # For batched/per-element schedules, enforce sigma_max at the first time index.
                t[..., 0] = sigma_max_tensor

    return t


def sample_timesteps_logsnr(batch_size, mean_logsnr=-1.2, std_logsnr=2.0):
    """
    Sample timesteps for diffusion training by sampling logSNR values and converting to t.

    Args:
        batch_size (int): Number of timesteps to sample
        mean_logsnr (float): Mean of the logSNR Gaussian distribution
        std_logsnr (float): Standard deviation of the logSNR Gaussian distribution

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing timestep values t in [0, 1]
    """
    # Sample logSNR from Gaussian distribution
    logsnr = torch.randn(batch_size) * std_logsnr + mean_logsnr

    # Convert logSNR to timesteps using the logistic function
    # Since logSNR = ln((1-t)/t), we can solve for t:
    # t = 1 / (1 + exp(logsnr))
    t = torch.sigmoid(-logsnr)

    # Clamp values to ensure numerical stability
    t = t.clamp(1e-4, 1 - 1e-4)

    return t

def sample_timesteps_logsnr_uniform(batch_size, min_logsnr=-6, max_logsnr=5.0):
    """
    Sample timesteps for diffusion training by sampling logSNR values and converting to t.

    Args:
        batch_size (int): Number of timesteps to sample
        min_logsnr (float): Minimum logSNR value
        max_logsnr (float): Maximum logSNR value

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing timestep values t in [0, 1]
    """
    # Sample logSNR from uniform distribution
    logsnr = torch.rand(batch_size) * (max_logsnr - min_logsnr) + min_logsnr

    # Convert logSNR to timesteps using the logistic function
    # Since logSNR = ln((1-t)/t), we can solve for t:
    # t = 1 / (1 + exp(logsnr))
    t = torch.sigmoid(-logsnr)

    # Clamp values to ensure numerical stability
    t = t.clamp(1e-4, 1 - 1e-4)

    return t

def truncated_logistic_normal_rescaled(shape, left_trunc=0.075, right_trunc=1):
    """

    shape: shape of the output tensor
    left_trunc: left truncation point, fraction of probability to be discarded
    right_trunc: right truncation boundary, should be 1 (never seen at test time)
    """

    # Step 1: Sample from the logistic normal distribution (sigmoid of normal)
    logits = torch.randn(shape)

    # Step 2: Apply the CDF transformation of the normal distribution
    normal_dist = dist.Normal(0, 1)
    cdf_values = normal_dist.cdf(logits)

    # Step 3: Define the truncation bounds on the CDF
    lower_bound = normal_dist.cdf(torch.logit(torch.tensor(left_trunc)))
    upper_bound = normal_dist.cdf(torch.logit(torch.tensor(right_trunc)))

    # Step 4: Rescale linear CDF values into the truncated region (between lower_bound and upper_bound)
    truncated_cdf_values = lower_bound + (upper_bound - lower_bound) * cdf_values

    # Step 5: Map back to logistic-normal space using inverse CDF
    truncated_samples = torch.sigmoid(normal_dist.icdf(truncated_cdf_values))

    # Step 6: Rescale values so that min is 0 and max is just below 1
    rescaled_samples = (truncated_samples - left_trunc) / (right_trunc - left_trunc)

    return rescaled_samples

def sample_discrete_euler(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. Euler method

    Args:
        sigmas: Pre-computed schedule tensor. Shape (steps+1,) for global schedule
            or (batch_size, steps+1) for per-element schedules.
    """
    t = sigmas

    # Check if we have per-element schedules (batch_size, steps+1) or global schedule (steps+1,)
    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        if per_element_schedule:
            # Per-element schedules: t has shape (batch_size, steps+1)
            t_curr_tensor = t[:, i].to(x.dtype)  # (batch_size,)
            t_prev = t[:, i + 1].to(x.dtype)  # (batch_size,)
            dt = t_prev - t_curr_tensor  # (batch_size,)
            # Reshape for broadcasting with x: (batch_size,) -> (batch_size, 1, 1)
            dt_broadcast = dt.view(-1, 1, 1)
        else:
            # Global schedule: t has shape (steps+1,)
            t_curr = t[i]
            t_prev = t[i + 1]
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
            dt = t_prev - t_curr
            dt_broadcast = dt

        v = model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            denoised = x - t_curr_tensor[:, None, None] * v
            callback({'x': x, 't': t_curr_tensor, 'sigma': t_curr_tensor, 'i': i, 'denoised': denoised})

        x = x + dt_broadcast * v

    # If we are on the last timestep, output the denoised data
    return x

def sample_rk4(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. 4th-order Runge-Kutta

    Args:
        sigmas: Pre-computed schedule tensor of shape (steps+1,).
            Per-element schedules not supported for RK4.
    """
    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    t = sigmas

    t = t.to(x.device)

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]), disable=disable_tqdm)):
        # Broadcast the current timestep to the correct shape
        t_curr_tensor = t_curr * ts
        dt = t_prev - t_curr  # we solve backwards in our formulation

        k1 = model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            denoised = x - t_curr * k1
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i, 'denoised': denoised})

        k2 = model(x + dt / 2 * k1, (t_curr + dt / 2) * ts, **extra_args)
        k3 = model(x + dt / 2 * k2, (t_curr + dt / 2) * ts, **extra_args)

        # Clamp t_prev to avoid evaluating model at exactly t=0
        # (models aren't trained at t=0 and may return garbage/NaN)
        t_prev_eval = t_prev.clamp(min=1e-5)
        k4 = model(x + dt * k3, t_prev_eval * ts, **extra_args)

        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # If we are on the last timestep, output the denoised data
    return x

def sample_flow_dpmpp(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. DPM-Solver++ for RF models

    Args:
        sigmas: Pre-computed schedule tensor. Shape (steps+1,) for global schedule
            or (batch_size, steps+1) for per-element schedules.
    """
    t = sigmas

    # Check if we have per-element schedules (batch_size, steps+1) or global schedule (steps+1,)
    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    old_denoised = None

    # Clamp t to avoid numerical issues with log(0) and division by zero
    # This prevents inf/-inf values that can cause NaN propagation
    log_snr = lambda t: ((1-t).clamp(min=1e-10) / t.clamp(min=1e-10)).log()

    for i in trange(num_steps, disable=disable_tqdm):
        if per_element_schedule:
            # Per-element schedules: t has shape (batch_size, steps+1)
            t_curr = t[:, i]  # (batch_size,)
            t_next = t[:, i + 1]  # (batch_size,)
            t_prev = t[:, i - 1] if i > 0 else None
            # Reshape for broadcasting with x: (batch_size,) -> (batch_size, 1, 1)
            t_curr_broadcast = t_curr.view(-1, 1, 1)
            t_next_broadcast = t_next.view(-1, 1, 1)
            t_curr_tensor = t_curr  # already (batch_size,)
        else:
            # Global schedule: t has shape (steps+1,)
            t_curr = t[i]
            t_next = t[i + 1]
            t_prev = t[i - 1] if i > 0 else None
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next
            t_curr_tensor = t_curr.expand(x.shape[0])

        model_output = model(x, t_curr_tensor, **extra_args)
        denoised = x - t_curr_broadcast * model_output

        if callback is not None:
            callback({'x': x, 'i': i, 't': t_curr, 'sigma': t_curr, 'denoised': denoised})

        alpha_t = 1 - t_next_broadcast

        # For rectified flow, compute the DPM++ coefficient directly without log_snr
        # to avoid numerical issues at t=0 or t=1
        # The formula is: (-h).expm1() = (t_next - t_curr) / [(1 - t_next) * t_curr]
        # Note: t_next < t_curr, so this is negative
        # We'll compute this directly instead of going through log_snr
        dt = t_next_broadcast - t_curr_broadcast
        # Clamp to avoid division by zero when t_curr or t_next are at boundaries
        dpmpp_coeff = dt / ((1 - t_next_broadcast).clamp(min=1e-10) * t_curr_broadcast.clamp(min=1e-10))

        # Check if this is the first step or the last step (t_next == 0)
        is_first_step = old_denoised is None
        is_last_step = (t_next_broadcast == 0).all() if per_element_schedule else (t_next == 0)

        if is_first_step or is_last_step:
            # First-order update using the directly computed coefficient
            x = (t_next_broadcast / t_curr_broadcast.clamp(min=1e-10)) * x - alpha_t * dpmpp_coeff * denoised
        else:
            # Second-order update with Richardson extrapolation
            if per_element_schedule:
                t_prev_broadcast = t_prev.view(-1, 1, 1)
            else:
                t_prev_broadcast = t_prev
            # Compute r = h_last / h in log-SNR space for second-order correction
            # h = log_snr(t_next) - log_snr(t_curr), h_last = log_snr(t_curr) - log_snr(t_prev)
            h = log_snr(t_next_broadcast) - log_snr(t_curr_broadcast)
            h_last = log_snr(t_curr_broadcast) - log_snr(t_prev_broadcast)
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_next_broadcast / t_curr_broadcast.clamp(min=1e-10)) * x - alpha_t * dpmpp_coeff * denoised_d

        old_denoised = denoised
    return x

def sample_flow_pingpong(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. Ping-pong sampling for distilled models

    Args:
        sigmas: Pre-computed schedule tensor. Shape (steps+1,) for global schedule
            or (batch_size, steps+1) for per-element schedules.
    """
    t = sigmas

    # Check if we have per-element schedules (batch_size, steps+1) or global schedule (steps+1,)
    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    for i in trange(num_steps, disable=disable_tqdm):
        if per_element_schedule:
            # Per-element schedules: t has shape (batch_size, steps+1)
            t_curr = t[:, i].to(x.dtype)  # (batch_size,)
            t_next = t[:, i + 1].to(x.dtype)  # (batch_size,)
            # Reshape for broadcasting with x: (batch_size,) -> (batch_size, 1, 1)
            t_curr_broadcast = t_curr.view(-1, 1, 1)
            t_next_broadcast = t_next.view(-1, 1, 1)
        else:
            # Global schedule: t has shape (steps+1,)
            t_curr = t[i].to(x.dtype)
            t_next = t[i + 1].to(x.dtype)
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next

        # Model forward
        if per_element_schedule:
            t_curr_tensor = t_curr  # already (batch_size,)
        else:
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)

        denoised = x - t_curr_broadcast * model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 't': t_curr, 'sigma': t_curr, 'sigma_hat': t_curr, 'denoised': denoised})

        x = (1 - t_next_broadcast) * denoised + t_next_broadcast * torch.randn_like(x)

    return x


def sample_v(model, x, sigmas, eta=0, callback=None, cfg_pp=False, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. v-diffusion DDIM.

    Args:
        sigmas: Pre-computed schedule tensor of shape (steps,).
    """
    ts = x.new_ones([x.shape[0]])

    t = sigmas.to(x.device)
    steps = len(t)
    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps, disable=disable_tqdm):

        if cfg_pp:
            # Get the model output (v, the predicted velocity)
            v, info = model(x, ts * t[i], return_info=True, **extra_args)

            if "uncond_output" in info:
                v_eps = info["uncond_output"]
            else:
                v_eps = v
        else:
            v = model(x, ts * t[i], **extra_args)
            v_eps = v

        # Predict the noise and the denoised data
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v_eps * alphas[i]

        if callback is not None:
            callback({'x': x, 't': t[i], 'sigma': sigmas[i], 'i': i, 'denoised': pred})

        # If we are not on the last timestep, compute the noisy data for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised data in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised data
    return pred

# Soft mask inpainting is just shrinking hard (binary) mask inpainting
# Given a float-valued soft mask (values between 0 and 1), get the binary mask for this particular step
def get_bmask(i, steps, mask):
    strength = (i+1)/(steps)
    # convert to binary mask
    bmask = torch.where(mask<=strength,1,0)
    return bmask

def make_cond_model_fn(model, cond_fn):
    def cond_model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return cond_model_fn

# Uses k-diffusion from https://github.com/crowsonkb/k-diffusion
# init_data is init_audio as latents (if this is latent diffusion)
# For sampling, init_data to none
# For variations, set init_data
def sample_k(
        model_fn,
        noise,
        init_data=None,
        steps=100,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.01,
        sigma_max=100,
        rho=1.0,
        device="cuda",
        callback=None,
        cond_fn=None,
        **extra_args
    ):

    is_k_diff = sampler_type in ["k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde","dpmpp-2m"]
    is_v_diff = sampler_type in ["v-ddim", "v-ddim-cfgpp"]

    if is_k_diff:

        denoiser = K.external.VDenoiser(model_fn)

        if cond_fn is not None:
            denoiser = make_cond_model_fn(denoiser, cond_fn)

        # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
        sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
        # Scale the initial noise by sigma
        noise = noise * sigmas[0]

        if init_data is not None:
            # set the initial latent to the init_data, and noise it with initial sigma
            x = init_data + noise
        else:
            # SAMPLING
            # set the initial latent to noise
            x = noise


        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m":
            return K.sampling.sample_dpmpp_2m(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-3m-sde":
            return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
    elif is_v_diff:

        if sigma_max > 1: # sigma_max should be between 0 and 1
            sigma_max = 1

        if cond_fn is not None:
            model_fn = make_cond_model_fn(model_fn, cond_fn)

        alpha, sigma = t_to_alpha_sigma(torch.tensor(sigma_max))

        if init_data is not None:
            x = init_data * alpha + noise * sigma
        else:
            x = noise

        if sampler_type == "v-ddim" or sampler_type == "v-ddim-cfgpp":
            use_cfg_pp = sampler_type == "v-ddim-cfgpp"
            t = build_schedule(steps=steps, sigma_max=sigma_max, include_endpoint=False, device=x.device)
            return sample_v(model_fn, x, sigmas=t, eta=0.0, cfg_pp=use_cfg_pp, callback=callback, **extra_args)
    else:
        raise ValueError(f"Unknown sampler type {sampler_type}")


@torch.no_grad()
def sample_diffusion(
    model,
    noise: torch.Tensor,
    cond_inputs: dict,
    diffusion_objective: str,
    steps: int,
    cfg_scale: float = 1.0,
    # Varlen support
    conditioning: tp.Optional[tp.List[dict]] = None,
    sample_rate: int = 44100,
    pretransform = None,
    mask_padding_attention: bool = False,
    use_effective_length_for_schedule: bool = False,
    headroom_seconds: float = 5.0,
    padding_mask: tp.Optional[torch.Tensor] = None,
    # Timestep schedule
    dist_shift = None,
    # Sampler options
    sampler_type: str = None,
    batch_cfg: bool = True,
    rescale_cfg: bool = False,
    # CFG options
    apg_scale: float = 1.0,
    # Init data (variation / img2img)
    init_data: tp.Optional[torch.Tensor] = None,
    init_noise_level: float = 1.0,
    # Other
    callback = None,
    disable_tqdm: bool = False,
    decode: bool = True,
    **sampler_kwargs
) -> torch.Tensor:
    """
    Unified sampling function for diffusion models. Handles all diffusion objectives,
    varlen support (padding_mask + effective_seq_len), timestep scheduling, and init_data
    for variation/img2img.

    Args:
        model: The diffusion model backbone (model.model, not the wrapper)
        noise: Initial noise tensor of shape (B, C, T)
        cond_inputs: Pre-processed conditioning inputs dict (merged positive + negative)
        diffusion_objective: One of "v", "rectified_flow", "rf_denoiser"
        steps: Number of sampling steps
        cfg_scale: Classifier-free guidance scale
        conditioning: List of conditioning dicts (for computing varlen from seconds_total)
        sample_rate: Audio sample rate
        pretransform: Optional pretransform for decoding latents and computing downsampling_ratio
        mask_padding_attention: Whether to create padding_mask for attention
        use_effective_length_for_schedule: Whether to use effective_seq_len for dist_shift
        padding_mask: Optional pre-computed padding mask (B, T). If provided, skips
            internal mask computation. Use this to ensure consistency with training masks.
        headroom_seconds: Extra seconds beyond seconds_total for valid region
        dist_shift: Distribution shift object for warping the timestep schedule, or None
        sampler_type: Sampler type. For RF: "euler", "rk4", "dpmpp", "pingpong".
            For v-diffusion: "v-ddim", "v-ddim-cfgpp", or k-diffusion types like "dpmpp-2m-sde".
        batch_cfg: Whether to use batched CFG
        rescale_cfg: Whether to use rescaled CFG
        apg_scale: APG (Adaptive Projected Guidance) scale. 1.0 = full APG, 0.0 = vanilla CFG
        init_data: Optional pre-encoded latent tensor for variation/img2img (shape: B, C, T)
        init_noise_level: Noise level (sigma_max) when using init_data. 1.0 = full noise (no variation).
        callback: Optional callback for progress reporting
        disable_tqdm: Whether to disable progress bar
        decode: Whether to decode latents using pretransform
        **sampler_kwargs: Additional kwargs passed to sampler

    Returns:
        Generated samples (decoded audio if decode=True, else latents)
    """
    device = noise.device
    batch_size = noise.shape[0]
    latent_seq_len = noise.shape[-1]

    # Compute downsampling ratio
    downsampling_ratio = pretransform.downsampling_ratio if pretransform is not None else 1

    # Default sampler_type per objective
    if sampler_type is None:
        sampler_type = "pingpong" if diffusion_objective == "rf_denoiser" else "euler"


    # Compute effective_seq_len for dist_shift if enabled
    effective_seq_len = None
    if use_effective_length_for_schedule and conditioning is not None:
        effective_seq_len = compute_effective_seq_len_from_conditioning(
            conditioning, sample_rate, downsampling_ratio, device
        )

    # Create padding_mask for attention if enabled (skip if pre-computed mask provided)
    if padding_mask is None and mask_padding_attention and conditioning is not None:
        raw_effective_len = compute_effective_seq_len_from_conditioning(
            conditioning, sample_rate, downsampling_ratio, device
        )
        if raw_effective_len is not None:
            headroom_tokens = int(headroom_seconds * sample_rate / downsampling_ratio)
            valid_lengths = (raw_effective_len + headroom_tokens).clamp(max=latent_seq_len).long()
            padding_mask = create_padding_mask_from_lengths(valid_lengths, latent_seq_len)

    # Determine sigma_max for schedule
    sigma_max = init_noise_level if init_data is not None else sampler_kwargs.get("sigma_max", 1.0)

    # Mix init_data with noise for variation/img2img
    # For k-diffusion v-diffusion samplers, init_data is passed through to sample_k
    # which handles mixing internally with its own sigma scaling
    k_diff_sampler_types = {"k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2",
                            "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m"}

    if init_data is not None:
        if diffusion_objective == "v" and sampler_type not in k_diff_sampler_types:
            # v-diffusion DDIM: pre-mix noise and init_data
            alpha, sigma = t_to_alpha_sigma(torch.tensor(sigma_max))
            noise = init_data * alpha + noise * sigma
        elif diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            # RF objectives: linear interpolation
            noise = init_data * (1 - sigma_max) + noise * sigma_max

    # Build common sampler kwargs (conditioning + model-level params only).
    # disable_tqdm and callback are passed explicitly to samplers that use them,
    # not included here, to avoid leaking into model forward() calls.
    common_kwargs = {
        **cond_inputs,
        "cfg_scale": cfg_scale,
        "batch_cfg": batch_cfg,
        "rescale_cfg": rescale_cfg,
        "padding_mask": padding_mask,
        "apg_scale": apg_scale,
        **sampler_kwargs
    }

    # Sample based on diffusion objective
    if diffusion_objective == "v":
        if sampler_type in k_diff_sampler_types or sampler_type in ["v-ddim", "v-ddim-cfgpp"]:
            # Route through sample_k which handles k-diffusion and v-ddim samplers
            # sample_k uses its own schedule (polyexponential for k-diff, internal for v-ddim)
            k_init_data = init_data if sampler_type in k_diff_sampler_types else None

            # Determine sigma_max for sample_k:
            # - k-diffusion samplers: default 100 for schedule, or init_noise_level for variations
            # - v-ddim: sigma_max is the noise level (0-1), our sigma_max variable
            if sampler_type in k_diff_sampler_types:
                k_sigma_max = sigma_max if init_data is not None else common_kwargs.pop("sigma_max", 100)
            else:
                k_sigma_max = sigma_max  # v-ddim: already 0-1 range
            # Pop sigma_max from common_kwargs to avoid passing it twice
            common_kwargs.pop("sigma_max", None)

            sampled = sample_k(
                model, noise,
                init_data=k_init_data,
                steps=steps,
                sampler_type=sampler_type,
                sigma_min=common_kwargs.pop("sigma_min", 0.01),
                sigma_max=k_sigma_max,
                rho=common_kwargs.pop("rho", 1.0),
                device=device,
                callback=callback,
                **common_kwargs
            )
        else:
            # DDIM-style sampler with pre-computed schedule
            t = build_schedule(
                steps=steps, sigma_max=sigma_max,
                dist_shift=dist_shift, effective_seq_len=effective_seq_len,
                fallback_seq_len=latent_seq_len, include_endpoint=False, device=device
            )
            sampled = sample_v(model, noise, sigmas=t, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)

    elif diffusion_objective in ["rectified_flow", "rf_denoiser"]:
        # Remove v-diffusion-specific kwargs that don't apply to RF
        common_kwargs.pop("sigma_min", None)
        common_kwargs.pop("sigma_max", None)
        common_kwargs.pop("rho", None)

        # Build schedule
        sigmas = build_schedule(
            steps=steps, sigma_max=sigma_max,
            dist_shift=dist_shift, effective_seq_len=effective_seq_len,
            fallback_seq_len=latent_seq_len, include_endpoint=True, device=device
        )

        # Route to sampler
        if sampler_type == "euler":
            sampled = sample_discrete_euler(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        elif sampler_type == "rk4":
            sampled = sample_rk4(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        elif sampler_type == "dpmpp":
            sampled = sample_flow_dpmpp(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        elif sampler_type == "pingpong":
            sampled = sample_flow_pingpong(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        else:
            raise ValueError(f"Unknown sampler_type for {diffusion_objective}: {sampler_type}")

    else:
        raise ValueError(f"Unknown diffusion_objective: {diffusion_objective}")

    # Decode if requested
    if decode and pretransform is not None:
        sampled = sampled.to(next(pretransform.parameters()).dtype)
        sampled = pretransform.decode(sampled)

        # Zero out audio beyond valid region (padding positions decode to garbage)
        if padding_mask is not None:
            audio_mask = padding_mask.unsqueeze(1).repeat_interleave(downsampling_ratio, dim=-1)
            # Trim or pad to match sampled length
            if audio_mask.shape[-1] > sampled.shape[-1]:
                audio_mask = audio_mask[..., :sampled.shape[-1]]
            elif audio_mask.shape[-1] < sampled.shape[-1]:
                audio_mask = torch.nn.functional.pad(audio_mask, (0, sampled.shape[-1] - audio_mask.shape[-1]), value=False)
            sampled = sampled * audio_mask.to(sampled.dtype)

    return sampled

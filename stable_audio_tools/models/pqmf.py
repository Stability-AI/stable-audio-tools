import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.optimize import fmin
from scipy.signal import firwin, kaiser, kaiser_beta, kaiserord

class PQMF(nn.Module):
    """
    Pseudo Quadrature Mirror Filter (PQMF) for multiband signal decomposition and reconstruction.
    Uses polyphase representation which is computationally more efficient for real-time. 
    
    Parameters:
    - attenuation (int): Desired attenuation of the rejected frequency bands, usually between 80 and 120 dB.
    - num_bands (int): Number of desired frequency bands. It must be a power of 2.
    """

    def __init__(self, attenuation, num_bands):
        super(PQMF, self).__init__()
        
        # Ensure num_bands is a power of 2 
        is_power_of_2 = (math.log2(num_bands) == int(math.log2(num_bands)))
        assert is_power_of_2, "'num_bands' must be a power of 2."
        
        # Create the prototype filter
        prototype_filter = design_prototype_filter(attenuation, num_bands)
        filter_bank = generate_modulated_filter_bank(prototype_filter, num_bands)
        padded_filter_bank = pad_to_nearest_power_of_two(filter_bank)
        
        # Register filters and settings
        self.register_buffer("filter_bank", padded_filter_bank)
        self.register_buffer("prototype", prototype_filter)
        self.num_bands = num_bands

    def forward(self, signal):
        """Decompose the signal into multiple frequency bands."""
        # If signal is not a pytorch tensor of Batch x Channels x Length, convert it 
        signal = prepare_signal_dimensions(signal)
        # The signal length must be a multiple of num_bands. Pad it with zeros.
        signal = pad_signal(signal, self.num_bands)
        # run it
        signal = polyphase_analysis(signal, self.filter_bank)
        return apply_alias_cancellation(signal)

    def inverse(self, bands):
        """Reconstruct the original signal from the frequency bands."""
        bands = apply_alias_cancellation(bands)
        return polyphase_synthesis(bands, self.filter_bank)


def prepare_signal_dimensions(signal):
    """
    Rearrange signal into Batch x Channels x Length. 
    
    Parameters
    ----------
    signal : torch.Tensor or numpy.ndarray
        The input signal.
        
    Returns
    -------
    torch.Tensor
        Preprocessed signal tensor.
    """
    # Convert numpy to torch tensor
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal)
        
    # Ensure tensor
    if not isinstance(signal, torch.Tensor):
        raise ValueError("Input should be either a numpy array or a PyTorch tensor.")
    
    # Modify dimension of signal to Batch x Channels x Length
    if signal.dim() == 1:
        # This is just a mono signal. Unsqueeze to 1 x 1 x Length
        signal = signal.unsqueeze(0).unsqueeze(0)
    elif signal.dim() == 2:
        # This is a multi-channel signal (e.g. stereo)
        # Rearrange so that larger dimension (Length) is last
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T
        # Unsqueeze to 1 x Channels x Length 
        signal = signal.unsqueeze(0)
    return signal
    
def pad_signal(signal, num_bands):
    """
    Pads the signal to make its length divisible by the given number of bands.

    Parameters
    ----------
    signal : torch.Tensor
        The input signal tensor, where the last dimension represents the signal length.

    num_bands : int
        The number of bands by which the signal length should be divisible.

    Returns
    -------
    torch.Tensor
        The padded signal tensor. If the original signal length was already divisible
        by num_bands, returns the original signal unchanged.
    """
    remainder = signal.shape[-1] % num_bands
    if remainder > 0:
        padding_size = num_bands - remainder
        signal = nn.functional.pad(signal, (0, padding_size))
    return signal

def generate_modulated_filter_bank(prototype_filter, num_bands):
    """
    Generate a QMF bank of cosine modulated filters based on a given prototype filter. 
    
    Parameters
    ----------
    prototype_filter : torch.Tensor
        The prototype filter used as the basis for modulation.
    num_bands : int
        The number of desired subbands or filters.
    
    Returns
    -------
    torch.Tensor
        A bank of cosine modulated filters.
    """
    
    # Initialize indices for modulation.
    subband_indices = torch.arange(num_bands).reshape(-1, 1)
    
    # Calculate the length of the prototype filter.
    filter_length = prototype_filter.shape[-1]
    
    # Generate symmetric time indices centered around zero.
    time_indices = torch.arange(-(filter_length // 2), (filter_length // 2) + 1)
    
    # Calculate phase offsets to ensure orthogonality between subbands.
    phase_offsets = (-1)**subband_indices * np.pi / 4
    
    # Compute the cosine modulation function.
    modulation = torch.cos(
        (2 * subband_indices + 1) * np.pi / (2 * num_bands) * time_indices + phase_offsets
    )
    
    # Apply modulation to the prototype filter.
    modulated_filters = 2 * prototype_filter * modulation
    
    return modulated_filters


def design_kaiser_lowpass(angular_cutoff, attenuation, filter_length=None):
    """
    Design a lowpass filter using the Kaiser window.
    
    Parameters
    ----------
    angular_cutoff : float
        The angular frequency cutoff of the filter.
    attenuation : float
        The desired stopband attenuation in decibels (dB).
    filter_length : int, optional
        Desired length of the filter. If not provided, it's computed based on the given specs.
    
    Returns
    -------
    ndarray
        The designed lowpass filter coefficients.
    """
    
    estimated_length, beta = kaiserord(attenuation, angular_cutoff / np.pi)
    
    # Ensure the estimated length is odd.
    estimated_length = 2 * (estimated_length // 2) + 1
    
    if filter_length is None:
        filter_length = estimated_length
    
    return firwin(filter_length, angular_cutoff, window=('kaiser', beta), scale=False, nyq=np.pi)


def evaluate_filter_objective(angular_cutoff, attenuation, num_bands, filter_length):
    """
    Evaluate the filter's objective value based on the criteria from https://ieeexplore.ieee.org/document/681427
    
    Parameters
    ----------
    angular_cutoff : float
        Angular frequency cutoff of the filter.
    attenuation : float
        Desired stopband attenuation in dB.
    num_bands : int
        Number of bands for the multiband filter system.
    filter_length : int, optional
        Desired length of the filter.
    
    Returns
    -------
    float
        The computed objective (loss) value for the given filter specs.
    """
    
    filter_coeffs = design_kaiser_lowpass(angular_cutoff, attenuation, filter_length)
    convolved_filter = np.convolve(filter_coeffs, filter_coeffs[::-1], "full")
    
    return np.max(np.abs(convolved_filter[convolved_filter.shape[-1] // 2::2 * num_bands][1:]))


def design_prototype_filter(attenuation, num_bands, filter_length=None):
    """
    Design the optimal prototype filter for a multiband system given the desired specs.
    
    Parameters
    ----------
    attenuation : float
        The desired stopband attenuation in dB.
    num_bands : int
        Number of bands for the multiband filter system.
    filter_length : int, optional
        Desired length of the filter. If not provided, it's computed based on the given specs.
    
    Returns
    -------
    ndarray
        The optimal prototype filter coefficients.
    """
    
    optimal_angular_cutoff = fmin(lambda angular_cutoff: evaluate_filter_objective(angular_cutoff, attenuation, num_bands, filter_length), 
                                  1 / num_bands, disp=0)[0]
    
    prototype_filter = design_kaiser_lowpass(optimal_angular_cutoff, attenuation, filter_length)
    return torch.tensor(prototype_filter, dtype=torch.float32)

def pad_to_nearest_power_of_two(x):
    """
    Pads the input tensor 'x' on both sides such that its last dimension 
    becomes the nearest larger power of two.
    
    Parameters:
    -----------
    x : torch.Tensor
        The input tensor to be padded.
        
    Returns:
    --------
    torch.Tensor
        The padded tensor.
    """
    current_length = x.shape[-1]
    target_length = 2**math.ceil(math.log2(current_length))
    
    total_padding = target_length - current_length
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    
    return nn.functional.pad(x, (left_padding, right_padding))

def apply_alias_cancellation(x):
    """
    Applies alias cancellation by inverting the sign of every 
    second element of every second row, starting from the second 
    row's first element in a tensor.
    
    This operation helps ensure that the aliasing introduced in 
    each band during the decomposition will be counteracted during 
    the reconstruction.
    
    Parameters:
    -----------
    x : torch.Tensor
        The input tensor.
        
    Returns:
    --------
    torch.Tensor
        Tensor with specific elements' sign inverted for alias cancellation.
    """
    
    # Create a mask of the same shape as 'x', initialized with all ones
    mask = torch.ones_like(x)
    
    # Update specific elements in the mask to -1 to perform inversion
    mask[..., 1::2, ::2] = -1
    
    # Apply the mask to the input tensor 'x'
    return x * mask

def ensure_odd_length(tensor):
    """
    Pads the last dimension of a tensor to ensure its size is odd.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        Input tensor whose last dimension might need padding.
        
    Returns:
    --------
    torch.Tensor
        The original tensor if its last dimension was already odd, 
        or the padded tensor with an odd-sized last dimension.
    """
    
    last_dim_size = tensor.shape[-1]
    
    if last_dim_size % 2 == 0:
        tensor = nn.functional.pad(tensor, (0, 1))
    
    return tensor

def polyphase_analysis(signal, filter_bank):
    """
    Applies the polyphase method to efficiently analyze the signal using a filter bank.

    Parameters:
    -----------
    signal : torch.Tensor
        Input signal tensor with shape (Batch x Channels x Length).
    
    filter_bank : torch.Tensor
        Filter bank tensor with shape (Bands x Length).
    
    Returns:
    --------
    torch.Tensor
        Signal split into sub-bands. (Batch x Channels x Bands x Length)
    """
    
    num_bands = filter_bank.shape[0]
    num_channels = signal.shape[1]
    
    # Rearrange signal for polyphase processing. 
    # Also combine Batch x Channel into one dimension for now. 
    #signal = rearrange(signal, "b c (t n) -> b (c n) t", n=num_bands)
    signal = rearrange(signal, "b c (t n) -> (b c) n t", n=num_bands)
    
    # Rearrange the filter bank for matching signal shape
    filter_bank = rearrange(filter_bank, "c (t n) -> c n t", n=num_bands)
    
    # Apply convolution with appropriate padding to maintain spatial dimensions
    padding = filter_bank.shape[-1] // 2
    filtered_signal = nn.functional.conv1d(signal, filter_bank, padding=padding)
    
    # Truncate the last dimension post-convolution to adjust the output shape
    filtered_signal = filtered_signal[..., :-1]
    # Rearrange the first dimension back into Batch x Channels
    filtered_signal = rearrange(filtered_signal, "(b c) n t -> b c n t", c=num_channels)

    return filtered_signal

def polyphase_synthesis(signal, filter_bank):
    """
    Polyphase Inverse: Apply polyphase filter bank synthesis to reconstruct a signal. 
    
    Parameters
    ----------
    signal : torch.Tensor
        Decomposed signal to be reconstructed (shape: Batch x Channels x Bands x Length).
    
    filter_bank : torch.Tensor
        Analysis filter bank (shape: Bands x Length).
    
    should_rearrange : bool, optional
        Flag to determine if the filters should be rearranged for polyphase synthesis. Default is True.

    Returns
    -------
    torch.Tensor
        Reconstructed signal (shape: Batch x Channels X Length)
    """
    
    num_bands = filter_bank.shape[0]
    num_channels = signal.shape[1]

    # Rearrange the filter bank 
    filter_bank = filter_bank.flip(-1)
    filter_bank = rearrange(filter_bank, "c (t n) -> n c t", n=num_bands)

    # Combine Batch x Channels into one dimension for now.
    signal = rearrange(signal, "b c n t -> (b c) n t")

    # Apply convolution with appropriate padding
    padding_amount = filter_bank.shape[-1] // 2 + 1
    reconstructed_signal = nn.functional.conv1d(signal, filter_bank, padding=int(padding_amount))
    
    # Scale the result
    reconstructed_signal = reconstructed_signal[..., :-1] * num_bands

    # Reorganize the output and truncate
    reconstructed_signal = reconstructed_signal.flip(1)
    reconstructed_signal = rearrange(reconstructed_signal, "(b c) n t -> b c (t n)", c=num_channels, n=num_bands)
    reconstructed_signal = reconstructed_signal[..., 2 * filter_bank.shape[1]:]
    
    return reconstructed_signal
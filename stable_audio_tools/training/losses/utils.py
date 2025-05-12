import torch
import math
from torch import nn
from einops import rearrange

class DynamicLossWeighting(nn.Module):
    def __init__(self, init_val = 1.0):
        super().__init__()
        self.loss_weight = nn.Parameter(torch.tensor(init_val))
    def forward(self, loss):
        return loss / torch.exp(self.loss_weight) + self.loss_weight

def flat_pairwise_sq_distance(x, y):
    """
    Compute pairwise squared Euclidean distances for flat tensors.
    Args:
        x: Tensor of shape (B, D)
        y: Tensor of shape (B, D)
    Returns:
        dist_sq: Tensor of shape (B, B)
    """
    x_norm = (x ** 2).mean(dim=1, keepdim=True)  # (B, 1)
    y_norm = (y ** 2).mean(dim=1, keepdim=True).transpose(0, 1)  # (1, B)
    return (x_norm + y_norm - 2.0 * torch.mm(x, y.t())/ x.shape[1] ) 

def multi_bandwidth_kernel_2d(x, y, bandwidths):
    """
    Compute the sum of Gaussian kernels (with different bandwidths) between two flat tensors.
    Args:
        x: Tensor of shape (B, D)
        y: Tensor of shape (B, D)
        bandwidths: Iterable of scalar bandwidth values.
    Returns:
        kernel: Tensor of shape (B, B)
    """
    dist_sq = flat_pairwise_sq_distance(x, y).clip(min = 0.0)
    kernel_sum = 0.0
    for bw in bandwidths:
        #kernel_sum += torch.exp(-dist_sq / (2.0 * bw))
        kernel_sum += (1/(1 + dist_sq / ( 2 * bw))).mean()
    return kernel_sum / len(bandwidths)

def mmd_loss_flat(x, y, bandwidths):
    """
    Compute the MMD loss between two flat sets of vectors.
    Args:
        x: Tensor of shape (B, D)
        y: Tensor of shape (B, D)
        bandwidths: Iterable of bandwidth values.
    Returns:
        loss: Scalar tensor representing the MMD loss.
    """
    K_xx = multi_bandwidth_kernel_2d(x, x, bandwidths)
    K_yy = multi_bandwidth_kernel_2d(y, y, bandwidths)
    K_xy = multi_bandwidth_kernel_2d(x, y, bandwidths)
    loss = K_xx + K_yy - 2.0 * K_xy
    return loss

def mmd(x, y, bandwidths =[1], dim = None):
    """
    Compute the MMD loss along a chosen feature axis by collapsing all other dimensions.
    
    Args:
        x: Tensor of arbitrary shape.
        y: Tensor of the same shape as x.
        bandwidths: Iterable of scalar bandwidth values for the kernel.
        dim: The axis index that should be treated as the feature dimension.
        
    Returns:
        Scalar tensor representing the MMD loss computed on the flattened representations.
    """
    if dim is None:
        dim_product = math.prod(x.shape[1:])
        new_shape = (-1, dim_product)
        x_flat = x.reshape(new_shape)
        y_flat = y.reshape(new_shape)
    else:
        dims = list(range(x.dim()))
        dims.pop(dim)
        dims.append(dim)
        x_perm = x.permute(*dims)
        y_perm = y.permute(*dims)
        # Collapse all dimensions except the last one.
        new_shape = (-1, x_perm.size(-1))
        x_flat = x_perm.reshape(new_shape)
        y_flat = y_perm.reshape(new_shape)
    return mmd_loss_flat(x_flat, y_flat, bandwidths)

def grouped_mmd(x, y, bandwidths = [1], groups = 2):
    grouped_x = rearrange(x, '... (g f) t -> ... g (f t)', g = groups)
    grouped_y = rearrange(y, '... (g f) t -> ... g (f t)', g = groups)
    return mmd(grouped_x, grouped_y, bandwidths, dim = None)
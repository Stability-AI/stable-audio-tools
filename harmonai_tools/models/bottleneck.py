import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange
from vector_quantize_pytorch import ResidualVQ
from nwt_pytorch import Memcodes

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x, return_info=False):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError
    
class TanhBottleneck(Bottleneck):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def encode(self, x, return_info=False):
        info = {}

        x = torch.tanh(x)

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x
    
class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__()

    def vae_sample(self, mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

    def encode(self, x, return_info=False):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = self.vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x
    
class L2Bottleneck(Bottleneck):
    def __init__(self):
        super().__init__()
    
    def encode(self, x, return_info=False):
        info = {}

        x = F.normalize(x, dim=1)

        if return_info:
            return x, info
        else:
            return x
        
    def decode(self, x):
        return F.normalize(x, dim=1)
        
class RVQBottleneck(Bottleneck):
    def __init__(self, **quantizer_kwargs):
        super().__init__()
        self.quantizer = ResidualVQ(**quantizer_kwargs)
        self.num_quantizers = quantizer_kwargs["num_quantizers"]

    def encode(self, x, return_info=False):
        info = {}

        x = rearrange(x, "b c n -> b n c")
        x, indices, loss = self.quantizer(x)
        x = rearrange(x, "b n c -> b c n")

        info["quantizer_indices"] = indices
        info["quantizer_loss"] = loss.mean()

        if return_info:
            return x, info
        else:
            return x
        
    def decode(self, x):
        return x
    
class MemcodesBottleneck(Bottleneck):
    def __init__(self, **memcodes_kwargs):
        super().__init__()
        self.quantizer = Memcodes(**memcodes_kwargs)

    def encode(self, x, return_info=False):
        info = {}

        x = rearrange(x, "b c n -> b n c")
        x, indices = self.quantizer(x)
        x = rearrange(x, "b n c -> b c n")

        info["quantizer_indices"] = indices

        if return_info:
            return x, info
        else:
            return x
        
    def decode(self, x):
        return x
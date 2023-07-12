import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange
from vector_quantize_pytorch import ResidualVQ
from nwt_pytorch import Memcodes
from dac.nn.quantize import ResidualVectorQuantize as DACResidualVQ

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

def vae_sample(mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__()

    def encode(self, x, return_info=False):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

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
    
class RVQVAEBottleneck(Bottleneck):
    def __init__(self, **quantizer_kwargs):
        super().__init__()
        self.quantizer = ResidualVQ(**quantizer_kwargs)
        self.num_quantizers = quantizer_kwargs["num_quantizers"]

    def encode(self, x, return_info=False):
        info = {}

        x, kl = vae_sample(*x.chunk(2, dim=1))

        info["kl"] = kl

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

class DACRVQBottleneck(Bottleneck):
    def __init__(self, quantize_on_decode=False, **quantizer_kwargs):
        super().__init__()
        self.quantizer = DACResidualVQ(**quantizer_kwargs)
        self.num_quantizers = quantizer_kwargs["n_codebooks"]
        self.quantize_on_decode = quantize_on_decode

    def encode(self, x, return_info=False):
        info = {}

        info["pre_quantizer"] = x

        if self.quantize_on_decode:
            return x, info if return_info else x

        output = self.quantizer(x)

        output["vq/commitment_loss"] /= self.num_quantizers
        output["vq/codebook_loss"] /= self.num_quantizers

        info.update(output)

        if return_info:
            return output["z"], info
        
        return output["z"]
    
    def decode(self, x):

        if self.quantize_on_decode:
            x = self.quantizer(x)["z"]

        return x

class DACRVQVAEBottleneck(Bottleneck):
    def __init__(self, quantize_on_decode=False, **quantizer_kwargs):
        super().__init__()
        self.quantizer = DACResidualVQ(**quantizer_kwargs)
        self.num_quantizers = quantizer_kwargs["n_codebooks"]
        self.quantize_on_decode = quantize_on_decode

    def encode(self, x, return_info=False):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["pre_quantizer"] = x
        info["kl"] = kl

        if self.quantize_on_decode:
            return x, info if return_info else x

        output = self.quantizer(x)

        output["vq/commitment_loss"] /= self.num_quantizers
        output["vq/codebook_loss"] /= self.num_quantizers

        info.update(output)

        if return_info:
            return output["z"], info
        
        return output["z"]
    
    def decode(self, x):

        if self.quantize_on_decode:
            x = self.quantizer(x)["z"]

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
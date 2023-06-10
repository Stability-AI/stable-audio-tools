import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange
from encodec.quantization.core_vq import ResidualVectorQuantization

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
        
class QuantizerBottleneck(Bottleneck):
    def __init__(self, quantizer_dropout=False, **quantizer_kwargs):
        super().__init__()
        self.quantizer = ResidualVectorQuantization(**quantizer_kwargs)
        self.num_quantizers = quantizer_kwargs["num_quantizers"]
        self.quantizer_dropout = quantizer_dropout

    def encode(self, x, return_info=False):
        info = {}

        n_q = self.num_quantizers

        if self.training and self.quantizer_dropout:
            n_q = int(torch.randint(1, self.num_quantizers + 1, (1,)).item())

        #x = rearrange(x, "b c n -> b n c")
        x, indices, loss = self.quantizer(x, n_q=n_q)
        #x = rearrange(x, "b n c -> b c n")

        info["quantizer_indices"] = indices
        info["quantizer_loss"] = loss.mean()

        if return_info:
            return x, info
        else:
            return x
        
    def decode(self, x):
        return x
    
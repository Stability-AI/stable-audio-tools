from torch import nn

class Pretransform(nn.Module):
    def __init__(self, enable_grad=False, io_channels=2):
        super().__init__()

        self.io_channels = io_channels
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def encode(self, x):
        return x

    def decode(self, z):
        return z

class AutoencoderPretransform(Pretransform):
    def __init__(self, model, scale=1.0):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale=scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
    
    def encode(self, x):
        encoded = self.model.encode(x)
        return encoded / self.scale

    def decode(self, z):
        z = z * self.scale
        return self.model.decode(z)
    
    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

class WaveletPretransform(Pretransform):
    def __init__(self, channels, levels, wavelet):
        super().__init__()

        from .wavelets import WaveletEncode1d, WaveletDecode1d

        self.encoder = WaveletEncode1d(channels, levels, wavelet)
        self.decoder = WaveletDecode1d(channels, levels, wavelet)

        self.downsampling_ratio = 2 ** levels
        self.io_channels = channels
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    

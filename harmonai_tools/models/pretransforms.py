from torch import nn

class Pretransform(nn.Module):
    def __init__(self, enable_grad=False, io_channels=2, ):
        super().__init__()

        self.io_channels = io_channels
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def encode(self, x):
        return x

    def decode(self, z):
        return z

class AutoencoderPretransform(Pretransform):
    def __init__(self, model, scale=1.0, model_half=False):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale=scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate
        
        self.model_half = model_half

        if self.model_half:
            self.model.half()
    
    def encode(self, x):
        
        if self.model_half:
            x = x.half()

        encoded = self.model.encode(x)

        if self.model_half:
            encoded = encoded.float()

        return encoded / self.scale

    def decode(self, z):
        z = z * self.scale

        if self.model_half:
            z = z.half()

        decoded = self.model.decode(z)

        if self.model_half:
            decoded = decoded.float()

        return decoded
    
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
    
class PretrainedDACPretransform(Pretransform):
    def __init__(self, model_type="44khz", model_bitrate="8kbps", scale=1.0, quantize_on_decode: bool = True, chunked=True):
        super().__init__()
        
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

    def encode(self, x):

        latents = self.model.encoder(x)

        if self.quantize_on_decode:
            output = latents
        else:
            z, _, latents, _, _ = self.model.quantizer(z, n_quantizers=self.model.n_codebooks)
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

        


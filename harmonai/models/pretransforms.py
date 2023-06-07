from torch import nn


class Pretransform(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsampling_ratio = None

    def encode(self, x):
        return x

    def decode(self, z):
        return z

class AutoencoderPretransform(Pretransform):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.downsampling_ratio = model.downsampling_ratio
    
    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)
    
    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)
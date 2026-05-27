import math
import torch
from torch import nn
from torch.nn.utils import weight_norm

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

class ConvNeXtBlock(nn.Module):

    def __init__(self, dim, kernel_size=7, mult=4, glu=False):
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = WNConv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)

        self.glu = glu

        if glu:
            self.proj_up = WNConv1d(dim, dim * mult * 2, kernel_size=1)
            self.act = nn.SiLU()
        else:
            self.proj_up = WNConv1d(dim, dim * mult, kernel_size=1)
            self.act = nn.GELU()

        self.proj_down = WNConv1d(dim * mult, dim, kernel_size=1)

        # Zero-init the last conv layer
        nn.init.zeros_(self.proj_down.weight)
        nn.init.zeros_(self.proj_down.bias)

        

    def forward(self, x):
        input = x
        x = self.dw_conv(x)
        x = self.proj_up(x)
        if self.glu:
            x, gate = x.chunk(2, dim=1)
            x = x * torch.sigmoid(gate)
        x = self.act(x)
        x = self.proj_down(x)

        return x + input

class ConvNextEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_blocks=3, conv_args = {}):
        super().__init__()

        self.layers = nn.ModuleList([ConvNeXtBlock(dim=in_channels, **conv_args) for _ in range(num_blocks)])

        self.downsample = WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2))

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x)
        
        x = self.downsample(x)

        return x

class ConvNextDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_blocks=3, conv_args = {}):
        super().__init__()

       
        self.upsample = WNConvTranspose1d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2))

        self.layers = nn.ModuleList([ConvNeXtBlock(dim=out_channels, **conv_args) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.upsample(x)
        for layer in self.layers:
            x = checkpoint(layer, x)
        return x

class ConvNeXtEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 num_blocks = None,
                 conv_args = {},
        ):
        super().__init__()
          
        c_mults = [1] + c_mults

        if num_blocks is None:
            num_blocks = [3] * (len(c_mults)-1)

        self.depth = len(c_mults)

        self.proj_in = WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)

        layers = []
        
        for i in range(self.depth-1):
            layers += [ConvNextEncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], conv_args=conv_args, num_blocks=num_blocks[i])]
        
        layers += [
            WNConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x, return_features=False):

        x = self.proj_in(x)

        return self.layers(x)


class ConvNeXtDecoder(nn.Module):
    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 conv_args = {}):
        super().__init__()

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3)
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers += [ConvNextDecoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i-1]*channels, stride=strides[i-1], conv_args=conv_args)]

        layers += [WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
import torch
from torch.nn import functional as F
from torch import nn

class DynamicLossWeighting(nn.Module):
    def __init__(self, init_val = 1.0):
        super().__init__()
        self.loss_weight = nn.Parameter(torch.tensor(init_val))
    def forward(self, loss):
        return loss / torch.exp(self.loss_weight) + self.loss_weight

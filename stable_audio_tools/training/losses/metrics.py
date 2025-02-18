import torch
import torchaudio

from torch.nn import functional as F
from torch import nn

### Metrics are loss-like functions that do not backpropagate gradients.

class PESQMetric(nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.resampler = (
            torchaudio.transforms.Resample(sample_rate, 16000)
            if sample_rate != 16000 else None)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        from pypesq import pesq
        
        if self.resampler is not None:
            inputs = self.resampler(inputs)
            targets = self.resampler(targets)

        inputs_np = inputs.cpu().numpy().astype("float64")
        targets_np = targets.cpu().numpy().astype("float64")
        batch_size = targets.shape[0]

        # Compute average pesq across batch size.
        val_pesq = (1.0 / batch_size) * sum(
            pesq(targets_np[i].reshape(-1), inputs_np[i].reshape(-1), 16000)
            for i in range(batch_size))
        return val_pesq
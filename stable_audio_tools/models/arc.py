import torch
import torch.nn.functional as F

from torch import nn

def get_relativistic_losses(score_real, score_fake):
    # Compute difference between real and fake scores
    diff = score_real - score_fake
    dis_loss = F.softplus(-diff).mean()
    gen_loss = F.softplus(diff).mean()
    return dis_loss, gen_loss

class ConvDiscriminator(nn.Module):
    def __init__(self, channels, soft_clip_scale=None, loss_type="lsgan"):
        super().__init__()

        self.loss_type = loss_type

        self.layers = nn.Sequential(
            nn.Conv1d(kernel_size=4, in_channels=channels, out_channels=channels, stride=2, padding=1), # x2 downsampling
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(),
            nn.Conv1d(kernel_size=4, in_channels=channels, out_channels=channels, stride=2, padding=1), # x4 downsampling
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(),
            nn.Conv1d(kernel_size=4, in_channels=channels, out_channels=channels, stride=2, padding=1), # x8 downsampling
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(),
            nn.Conv1d(kernel_size=4, in_channels=channels, out_channels=channels, stride=2, padding=1), # x16 downsampling
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(),
            nn.Conv1d(kernel_size=4, in_channels=channels, out_channels=1, stride=1, padding=0), # to 1 channel for score
        )

        self.soft_clip_scale = soft_clip_scale

    def forward(self, x):
        output = self.layers(x)

        if self.soft_clip_scale is not None:
            output = self.soft_clip_scale * torch.tanh(output/self.soft_clip_scale)

        return output

    def loss(self, reals, fakes, *args, **kwargs):
        real_scores = self(reals)
        fake_scores = self(fakes)

        loss_dis = loss_adv = 0

        if self.loss_type == "lsgan":
            # Calculate least-squares GAN losses
            loss_dis = torch.mean(fake_scores**2) + torch.mean ((1 - real_scores)**2)
            loss_adv = torch.mean((1 - fake_scores)**2)
        elif self.loss_type == "relativistic":

            diff = real_scores - fake_scores

            loss_dis = F.softplus(-diff).mean()
            loss_adv = F.softplus(diff).mean()

        return {
            "loss_dis": loss_dis,
            "loss_adv": loss_adv
        }

class ConvNeXtDiscriminator(nn.Module):
    def __init__(self, loss_type="lsgan", *args, **kwargs):
        super().__init__()

        from .convnext import ConvNeXtEncoder

        self.encoder = ConvNeXtEncoder(*args, **kwargs)

        self.loss_type = loss_type

    def forward(self, x):
        return self.encoder(x)

    def loss(self, reals, fakes, *args, **kwargs):
        real_scores = self(reals)
        fake_scores = self(fakes)

        loss_dis = loss_adv = 0

        if self.loss_type == "lsgan":
            # Calculate least-squares GAN losses
            loss_dis = torch.mean(fake_scores**2) + torch.mean ((1 - real_scores)**2)
            loss_adv = torch.mean((1 - fake_scores)**2)
        elif self.loss_type == "relativistic":
            
            diff = real_scores - fake_scores

            loss_dis = F.softplus(-diff).mean()
            loss_adv = F.softplus(diff).mean()

        return {
            "loss_dis": loss_dis,
            "loss_adv": loss_adv
        }
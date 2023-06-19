import torch
import torch.nn as nn
from encodec.msstftd import MultiScaleSTFTDiscriminator
from oobleck.discriminators import MultiDiscriminator, MultiScaleDiscriminator, MultiPeriodDiscriminator, SharedDiscriminatorConvNet
from functools import partial

def get_hinge_losses(score_real, score_fake):
    gen_loss = -score_fake.mean()
    dis_loss = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()
    return dis_loss, gen_loss

class OobleckDiscriminator(nn.Module):

    def __init__(
            self,
            in_channels=1,
            ):
        super().__init__()

        multi_scale_discriminator = partial(
            MultiScaleDiscriminator,
            n_scales=3,
            convnet=lambda: SharedDiscriminatorConvNet(
                in_size=in_channels,
                out_size=1, 
                capacity=32, 
                n_layers=4, 
                kernel_size=15, 
                stride=4, 
                activation=nn.SiLU,
                normalization=nn.utils.weight_norm,
                convolution=nn.Conv1d
            )
        )

        multi_period_discriminator = partial(
            MultiPeriodDiscriminator,
            periods=[2, 3, 5, 7, 11],
            convnet=lambda: SharedDiscriminatorConvNet(
                in_size=in_channels,
                out_size=1, 
                capacity=32, 
                n_layers=4, 
                kernel_size=15, 
                stride=4, 
                activation=nn.SiLU,
                normalization=nn.utils.weight_norm,
                convolution=nn.Conv2d
            )
        )

        self.multi_discriminator = MultiDiscriminator(
            [multi_scale_discriminator, multi_period_discriminator],
            ["reals", "fakes"]
        )

    def loss(self, reals, fakes):
        inputs = {
            "reals": reals,
            "fakes": fakes,
        }

        inputs = self.multi_discriminator(inputs)

        scores_real = inputs["score_reals"]
        scores_fake = inputs["score_fakes"]

        features_real = inputs["features_reals"]
        features_fake = inputs["features_fakes"]

        dis_loss, gen_loss = get_hinge_losses(scores_real, scores_fake)
         
        feature_matching_distance = torch.tensor(0.)

        for _, (scale_real, scale_fake) in enumerate(zip(features_real, features_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda real, fake: abs(real - fake).mean(),
                    scale_real,
                    scale_fake,
                )) / len(scale_real)
            
        return dis_loss, gen_loss, feature_matching_distance

class EncodecDiscriminator(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.discriminators = MultiScaleSTFTDiscriminator(*args, **kwargs)

    def forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def loss(self, x, y):
        feature_matching_distance = 0.
        logits_true, feature_true = self.forward(x)
        logits_fake, feature_fake = self.forward(y)

        dis_loss = torch.tensor(0.)
        adv_loss = torch.tensor(0.)

        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = get_hinge_losses(
                logits_true[i],
                logits_fake[i],
            )

            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv

        return dis_loss, adv_loss, feature_matching_distance

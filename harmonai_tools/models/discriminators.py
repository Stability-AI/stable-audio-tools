import torch
import torch.nn as nn
from encodec.msstftd import MultiScaleSTFTDiscriminator

class EncodecDiscriminator(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.discriminators = MultiScaleSTFTDiscriminator(*args, **kwargs)

    def forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def adversarial_combine(self, score_real, score_fake):
        loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
        loss_dis = loss_dis.mean()
        loss_gen = -score_fake.mean()
        return loss_dis, loss_gen

    def loss(self, x, y):
        feature_matching_distance = 0.
        logits_true, feature_true = self.forward(x)
        logits_fake, feature_fake = self.forward(y)

        loss_dis = 0
        loss_adv = 0

        pred_true = 0
        pred_fake = 0

        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = self.adversarial_combine(
                logits_true[i],
                logits_fake[i],
            )

            pred_true = pred_true + logits_true[i].mean()
            pred_fake = pred_fake + logits_fake[i].mean()

            loss_dis = loss_dis + _dis
            loss_adv = loss_adv + _adv

        return loss_dis, loss_adv, feature_matching_distance, pred_true, pred_fake

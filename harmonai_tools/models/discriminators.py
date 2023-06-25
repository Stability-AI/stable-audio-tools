import torch
import torch.nn as nn
import numpy as np
from encodec.msstftd import MultiScaleSTFTDiscriminator
from functools import reduce
import typing as tp

def get_hinge_losses(score_real, score_fake):
    gen_loss = -score_fake.mean()
    dis_loss = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()
    return dis_loss, gen_loss

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

# Discriminators from oobleck

IndividualDiscriminatorOut = tp.Tuple[torch.Tensor, tp.Sequence[torch.Tensor]]

TensorDict = tp.Dict[str, torch.Tensor]

class SharedDiscriminatorConvNet(nn.Module):

    def __init__(
        self,
        in_size: int,
        convolution: tp.Union[nn.Conv1d, nn.Conv2d],
        out_size: int = 1,
        capacity: int = 32,
        n_layers: int = 4,
        kernel_size: int = 15,
        stride: int = 4,
        activation: tp.Callable[[], nn.Module] = lambda: nn.SiLU(),
        normalization: tp.Callable[[nn.Module], nn.Module] = torch.nn.utils.weight_norm,
    ) -> None:
        super().__init__()
        channels = [in_size]
        channels += list(capacity * 2**np.arange(n_layers))

        if isinstance(stride, int):
            stride = n_layers * [stride]

        net = []
        for i in range(n_layers):
            if isinstance(kernel_size, int):
                pad = kernel_size // 2
                s = stride[i]
            else:
                pad = kernel_size[0] // 2
                s = (stride[i], 1)

            net.append(
                normalization(
                    convolution(
                        channels[i],
                        channels[i + 1],
                        kernel_size,
                        stride=s,
                        padding=pad,
                    )))
            net.append(activation())

        net.append(convolution(channels[-1], out_size, 1))

        self.net = nn.ModuleList(net)

    def forward(self, x) -> IndividualDiscriminatorOut:
        features = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.modules.conv._ConvNd):
                features.append(x)
        score = x.reshape(x.shape[0], -1).mean(-1)
        return score, features


class MultiScaleDiscriminator(nn.Module):

    def __init__(self,
                in_channels: int,
                n_scales: int,
                **conv_kwargs) -> None:
        super().__init__()
        layers = []
        for _ in range(n_scales):
            layers.append(SharedDiscriminatorConvNet(in_channels, nn.Conv1d, **conv_kwargs))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> IndividualDiscriminatorOut:
        score = 0
        features = []
        for layer in self.layers:
            s, f = layer(x)
            score = score + s
            features.extend(f)
            x = nn.functional.avg_pool1d(x, 2)
        return score, features

class MultiPeriodDiscriminator(nn.Module):

    def __init__(self,
                 in_channels: int,
                 periods: tp.Sequence[int],
                 **conv_kwargs) -> None:
        super().__init__()
        layers = []
        self.periods = periods

        for _ in periods:
            layers.append(SharedDiscriminatorConvNet(in_channels, nn.Conv2d, **conv_kwargs))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> IndividualDiscriminatorOut:
        score = 0
        features = []
        for layer, n in zip(self.layers, self.periods):
            s, f = layer(self.fold(x, n))
            score = score + s
            features.extend(f)
        return score, features

    def fold(self, x: torch.Tensor, n: int) -> torch.Tensor:
        pad = (n - (x.shape[-1] % n)) % n
        x = nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:2], -1, n)


class MultiDiscriminator(nn.Module):
    """
    Individual discriminators should take a single tensor as input (NxB C T) and
    return a tuple composed of a score tensor (NxB) and a Sequence of Features
    Sequence[NxB C' T'].
    """

    def __init__(self, discriminator_list: tp.Sequence[nn.Module],
                 keys: tp.Sequence[str]) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(discriminator_list)
        self.keys = keys

    def unpack_tensor_to_dict(self, features: torch.Tensor) -> TensorDict:
        features = features.chunk(len(self.keys), 0)
        return {k: features[i] for i, k in enumerate(self.keys)}

    @staticmethod
    def concat_dicts(dict_a, dict_b):
        out_dict = {}
        keys = set(list(dict_a.keys()) + list(dict_b.keys()))
        for k in keys:
            out_dict[k] = []
            if k in dict_a:
                if isinstance(dict_a[k], list):
                    out_dict[k].extend(dict_a[k])
                else:
                    out_dict[k].append(dict_a[k])
            if k in dict_b:
                if isinstance(dict_b[k], list):
                    out_dict[k].extend(dict_b[k])
                else:
                    out_dict[k].append(dict_b[k])
        return out_dict

    @staticmethod
    def sum_dicts(dict_a, dict_b):
        out_dict = {}
        keys = set(list(dict_a.keys()) + list(dict_b.keys()))
        for k in keys:
            out_dict[k] = 0.
            if k in dict_a:
                out_dict[k] = out_dict[k] + dict_a[k]
            if k in dict_b:
                out_dict[k] = out_dict[k] + dict_b[k]
        return out_dict

    def forward(self, inputs: TensorDict) -> TensorDict:
        discriminator_input = torch.cat([inputs[k] for k in self.keys], 0)
        all_scores = []
        all_features = []

        for discriminator in self.discriminators:
            score, features = discriminator(discriminator_input)
            scores = self.unpack_tensor_to_dict(score)
            scores = {f"score_{k}": scores[k] for k in scores.keys()}
            all_scores.append(scores)

            features = map(self.unpack_tensor_to_dict, features)
            features = reduce(self.concat_dicts, features)
            features = {f"features_{k}": features[k] for k in features.keys()}
            all_features.append(features)

        all_scores = reduce(self.sum_dicts, all_scores)
        all_features = reduce(self.concat_dicts, all_features)

        inputs.update(all_scores)
        inputs.update(all_features)

        return inputs
    
class OobleckDiscriminator(nn.Module):

    def __init__(
            self,
            in_channels=1,
            ):
        super().__init__()

        multi_scale_discriminator = MultiScaleDiscriminator(
            in_channels=in_channels,
            n_scales=3,
        )

        multi_period_discriminator = MultiPeriodDiscriminator(
            in_channels=in_channels,
            periods=[2, 3, 5, 7, 11]
        )

        # multi_resolution_discriminator = MultiScaleSTFTDiscriminator(
        #     filters=32,
        #     in_channels = in_channels,
        #     out_channels = 1,
        #     n_ffts = [2048, 1024, 512, 256, 128],
        #     hop_lengths = [512, 256, 128, 64, 32],
        #     win_lengths = [2048, 1024, 512, 256, 128]
        # )

        self.multi_discriminator = MultiDiscriminator(
            [multi_scale_discriminator, multi_period_discriminator], #, multi_resolution_discriminator],
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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing as tp

from typing import List, Tuple
from functools import reduce
from einops import rearrange

from .autoencoders import checkpoint


def get_hinge_losses(score_real, score_fake):
    gen_loss = -score_fake.mean()
    dis_loss = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()
    return dis_loss, gen_loss

class EncodecDiscriminator(nn.Module):
    def __init__(self, normalize_losses = False, *args, **kwargs):
        super().__init__()

        from .encodec import MultiScaleSTFTDiscriminator

        self.discriminators = MultiScaleSTFTDiscriminator(*args, **kwargs)
        self.normalize_losses = normalize_losses

        if self.normalize_losses:
            self.fm_reduction = lambda x, y: abs(x - y).mean()/(abs(x).mean() + 1e-3)
        else:
            self.fm_reduction = lambda x, y: abs(x - y).mean()

    def forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def loss(self, reals, fakes):
        feature_matching_distance = torch.tensor(0., device=reals.device)
        logits_true, feature_true = self.forward(reals)
        logits_fake, feature_fake = self.forward(fakes)
        dis_loss = torch.tensor(0.,device=reals.device)
        adv_loss = torch.tensor(0.,device=reals.device)
        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):
            
            feature_matching_distance = feature_matching_distance + sum(
                map(
                    self.fm_reduction,
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = get_hinge_losses(
                logits_true[i],
                logits_fake[i],
            ) 

            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv

        return dis_loss / len(logits_true), adv_loss / len(logits_true), feature_matching_distance / len(logits_true)


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
    

## Discriminators from Descript Audio Codec repo
## Copied and modified under MIT license, see LICENSES/LICENSE_DESCRIPT.txt
class MPD(nn.Module):
    def __init__(self, period, channels=1):
        super().__init__()

        from dac.model.discriminator import WNConv2d

        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(channels, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MSD(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 44100, channels=1):
        super().__init__()

        from dac.model.discriminator import WNConv1d

        self.convs = nn.ModuleList(
            [
                WNConv1d(channels, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        x = AudioSignal(x, self.sample_rate)
        x.resample(self.sample_rate // self.rate)
        x = x.audio_data

        fmap = []

        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
        channels: int = 1
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        from dac.model.discriminator import WNConv2d
        from audiotools import STFTParams

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        self.channels = channels

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        from audiotools import AudioSignal
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())
        x = rearrange(x, "b ch f t c -> (b ch) c t f", ch=self.channels)
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap

class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.cfg = cfg
        # Using get with defaults
        self.cfg["cqtd_filters"] = self.cfg.get("cqtd_filters", 32)
        self.cfg["cqtd_max_filters"] = self.cfg.get("cqtd_max_filters", 1024)
        self.cfg["cqtd_filters_scale"] = self.cfg.get("cqtd_filters_scale", 1)
        self.cfg["cqtd_dilations"] = self.cfg.get("cqtd_dilations", [1, 2, 4])
        self.cfg["cqtd_in_channels"] = self.cfg.get("cqtd_in_channels", 1)
        self.cfg["cqtd_out_channels"] = self.cfg.get("cqtd_out_channels", 1)
        # Multi-scale params to loop over
        self.cfg["cqtd_hop_lengths"] = self.cfg.get("cqtd_hop_lengths", [512, 256, 256])
        self.cfg["cqtd_n_octaves"] = self.cfg.get("cqtd_n_octaves", [9, 9, 9])
        self.cfg["cqtd_bins_per_octaves"] = self.cfg.get(
            "cqtd_bins_per_octaves", [24, 36, 48])
        self.cfg["cqtd_fmin"] = self.cfg.get("fmin", 32.7)

        n_discriminators = len(self.cfg["cqtd_hop_lengths"])
        self.discriminators = nn.ModuleList([DiscriminatorCQT(
            self.cfg,
            hop_length=self.cfg["cqtd_hop_lengths"][i],
            n_octaves=self.cfg["cqtd_n_octaves"][i],
            bins_per_octave=self.cfg["cqtd_bins_per_octaves"][i],
        ) for i in range(n_discriminators)])

    def forward(self, reals: torch.Tensor, gens: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = checkpoint(disc,reals)
            y_d_g, fmap_g = checkpoint(disc,gens)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def discriminator_loss(self, fake, real):
        y_real, y_fake, fmap_real, fmap_fake = self.forward(real, fake.clone().detach())

        loss_d = 0
        for x_fake, x_real in zip(y_fake, y_real):
            loss_d += torch.mean(x_fake ** 2)
            loss_d += torch.mean((1 - x_real) ** 2)
        loss_d /= len(y_fake)
        return loss_d

    def generator_loss(self, fake, real):
        y_real, y_fake, fmap_real, fmap_fake = self.forward(real, fake)

        loss_g = 0
        for x_fake in y_fake:
            loss_g += torch.mean((1 - x_fake) ** 2)

        counter = 0
        loss_feature = 0
        for i in range(len(fmap_fake)):
            for j in range(len(fmap_fake[i])):
                denominator = fmap_real[i][j].abs().mean().detach()
                loss_feature += F.l1_loss(fmap_fake[i][j], fmap_real[i][j].detach()) / denominator
                counter += 1
        loss_feature /= counter
        loss_g /= len(y_fake)
        return loss_g, loss_feature

    def loss(self, reals, fakes):
        gen_loss, feature_distance = self.generator_loss(fakes, reals)
        dis_loss = self.discriminator_loss(fakes, reals)
        return dis_loss, gen_loss, feature_distance

class DACDiscriminator(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 44100,
        bands: list = BANDS,
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        discs += [MPD(p, channels=channels) for p in periods]
        discs += [MSD(r, sample_rate=sample_rate, channels=channels) for r in rates]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands, channels=channels) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [checkpoint(d,x) for d in self.discriminators]
        return fmaps

class DACGANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, use_hinge: bool = False, **discriminator_kwargs):
        super().__init__()
        self.use_hinge = use_hinge
        self.discriminator = DACDiscriminator(**discriminator_kwargs)

    def forward(self, fake, real):
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += (
                F.relu(x_fake[-1]).mean() +
                F.relu(1 - x_real[-1]).mean()
            ) if self.use_hinge else (
                (x_fake[-1] ** 2).mean() +
                ((1 - x_real[-1]) ** 2).mean()
            )
        loss_d /= len(d_fake)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += (
                F.relu(1 - x_fake[-1]).mean()
                if self.use_hinge else
                ((1 - x_fake[-1]) ** 2).mean()
            )

        n_discriminators = len(d_fake)
        loss_feature = 0
        for i in range(n_discriminators):
            # Average over N model layers (except for the last item, which is logits).
            n_layers = len(d_fake[i]) - 1
            loss_feature += sum(map(
                lambda j: F.l1_loss(d_fake[i][j], d_real[i][j].detach()),
                range(n_layers)
            )) / n_layers

        # Average over K discriminators.
        loss_feature = loss_feature / n_discriminators

        loss_g /= len(d_fake)
        return loss_g, loss_feature

    def loss(self, reals, fakes):
        gen_loss, feature_distance = self.generator_loss(fakes, reals)
        dis_loss = self.discriminator_loss(fakes, reals)
        return dis_loss, gen_loss, feature_distance

class BigVGANDiscriminator(nn.Module):
    def __init__(self, sample_rate: int,
        channels: int = 1,
        use_hinge: bool = False,
        periods: List[int] = [2, 3, 5, 7, 11],
        **cqt_kwargs,
    ):
        super().__init__()

        # Use MPD discriminator from DAC GAN, disable others.
        self.mpd = DACGANLoss(use_hinge=use_hinge, sample_rate=sample_rate,
            periods=periods, rates=[], fft_sizes=[], channels = channels)

        self.cqt = MultiScaleSubbandCQTDiscriminator({
            "cqtd_in_channels": channels,
            "sampling_rate": sample_rate, **cqt_kwargs,
        })

    def loss(self, reals, fakes):
        cqt_dis_loss, cqt_gen_loss, cqt_feature_distance = self.cqt.loss(reals, fakes)
        mpd_dis_loss, mpd_gen_loss, mpd_feature_distance = self.mpd.loss(reals, fakes)
        return (
            mpd_dis_loss + cqt_dis_loss,
            mpd_gen_loss + cqt_gen_loss,
            mpd_feature_distance + cqt_feature_distance)

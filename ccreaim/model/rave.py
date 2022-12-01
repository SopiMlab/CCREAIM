import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as wn
from einops import rearrange
from sklearn.decomposition import PCA

from ccreaim.utils.cfg_classes import HyperConfig

from ..utils.pqmf import CachedPQMF as PQMF
from ..utils.rave_core import (
    Loudness,
    amp_to_impulse_response,
    fft_convolve,
    mod_sigmoid,
    multiscale_stft,
)

log = logging.getLogger(__name__)

import cached_conv as cc


class Residual(nn.Module):
    def __init__(self, module, cumulative_delay=0):
        super().__init__()
        additional_delay = module.cumulative_delay
        self.aligned = cc.AlignBranches(
            module,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res


class ResidualStack(nn.Module):
    def __init__(self, dim, kernel_size, padding_mode, cumulative_delay=0, bias=False):
        super().__init__()
        net = []

        res_cum_delay = 0
        # SEQUENTIAL RESIDUALS
        for i in range(3):
            # RESIDUAL BLOCK
            seq = [nn.LeakyReLU(0.2)]
            seq.append(
                wn(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        padding=cc.get_padding(
                            kernel_size,
                            dilation=3**i,
                            mode=padding_mode,
                        ),
                        dilation=3**i,
                        bias=bias,
                    )
                )
            )

            seq.append(nn.LeakyReLU(0.2))
            seq.append(
                wn(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        padding=cc.get_padding(kernel_size, mode=padding_mode),
                        bias=bias,
                        cumulative_delay=seq[-2].cumulative_delay,
                    )
                )
            )

            res_net = cc.CachedSequential(*seq)

            net.append(Residual(res_net, cumulative_delay=res_cum_delay))
            res_cum_delay = net[-1].cumulative_delay

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay

    def forward(self, x):
        return self.net(x)


class UpsampleLayer(nn.Module):
    def __init__(
        self, in_dim, out_dim, ratio, padding_mode, cumulative_delay=0, bias=False
    ):
        super().__init__()
        net = [nn.LeakyReLU(0.2)]
        if ratio > 1:
            net.append(
                wn(
                    cc.ConvTranspose1d(
                        in_dim,
                        out_dim,
                        2 * ratio,
                        stride=ratio,
                        padding=ratio // 2,
                        bias=bias,
                    )
                )
            )
        else:
            net.append(
                wn(
                    cc.Conv1d(
                        in_dim,
                        out_dim,
                        3,
                        padding=cc.get_padding(3, mode=padding_mode),
                        bias=bias,
                    )
                )
            )

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


class NoiseGenerator(nn.Module):
    def __init__(self, in_size, data_size, ratios, noise_bands, padding_mode):
        super().__init__()
        net = []
        channels = [in_size] * len(ratios) + [data_size * noise_bands]
        cum_delay = 0
        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=cc.get_padding(3, r, mode=padding_mode),
                    stride=r,
                    cumulative_delay=cum_delay,
                )
            )
            cum_delay = net[-1].cumulative_delay
            if i != len(ratios) - 1:
                net.append(nn.LeakyReLU(0.2))

        self.net = cc.CachedSequential(*net)
        self.data_size = data_size
        self.cumulative_delay = self.net.cumulative_delay * int(np.prod(ratios))

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise


class Generator(nn.Module):
    def __init__(
        self,
        latent_size,
        capacity,
        data_size,
        ratios,
        loud_stride,
        use_noise,
        noise_ratios,
        noise_bands,
        padding_mode,
        bias=False,
    ):
        super().__init__()
        net = [
            wn(
                cc.Conv1d(
                    latent_size,
                    2 ** len(ratios) * capacity,
                    7,
                    padding=cc.get_padding(7, mode=padding_mode),
                    bias=bias,
                )
            )
        ]

        for i, r in enumerate(ratios):
            in_dim = 2 ** (len(ratios) - i) * capacity
            out_dim = 2 ** (len(ratios) - i - 1) * capacity

            net.append(
                UpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    padding_mode,
                    cumulative_delay=net[-1].cumulative_delay,
                )
            )
            net.append(
                ResidualStack(
                    out_dim,
                    3,
                    padding_mode,
                    cumulative_delay=net[-1].cumulative_delay,
                )
            )

        self.net = cc.CachedSequential(*net)

        wave_gen = wn(
            cc.Conv1d(
                out_dim,
                data_size,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias,
            )
        )

        loud_gen = wn(
            cc.Conv1d(
                out_dim,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(
                    2 * loud_stride + 1, loud_stride, mode=padding_mode
                ),
                bias=bias,
            )
        )

        branches = [wave_gen, loud_gen]

        if use_noise:
            noise_gen = NoiseGenerator(
                out_dim,
                data_size,
                noise_ratios,
                noise_bands,
                padding_mode=padding_mode,
            )
            branches.append(noise_gen)

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )

        self.use_noise = use_noise
        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

    def forward(self, x, add_noise: bool = True):
        x = self.net(x)

        if self.use_noise:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)

        loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        if add_noise:
            waveform = waveform + noise

        return waveform


class Encoder(nn.Module):
    def __init__(
        self, data_size, capacity, latent_size, ratios, padding_mode, bias=False
    ):
        super().__init__()
        net = [
            cc.Conv1d(
                data_size,
                capacity,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias,
            )
        ]

        for i, r in enumerate(ratios):
            in_dim = 2**i * capacity
            out_dim = 2 ** (i + 1) * capacity

            net.append(nn.BatchNorm1d(in_dim))
            net.append(nn.LeakyReLU(0.2))
            net.append(
                cc.Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    padding=cc.get_padding(2 * r + 1, r, mode=padding_mode),
                    stride=r,
                    bias=bias,
                    cumulative_delay=net[-3].cumulative_delay,
                )
            )

        net.append(nn.LeakyReLU(0.2))
        net.append(
            cc.Conv1d(
                out_dim,
                2 * latent_size,
                5,
                padding=cc.get_padding(5, mode=padding_mode),
                groups=2,
                bias=bias,
                cumulative_delay=net[-2].cumulative_delay,
            )
        )

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        z = self.net(x)
        return torch.split(z, z.shape[1] // 2, 1)


class Discriminator(nn.Module):
    def __init__(self, in_size, capacity, multiplier, n_layers):
        super().__init__()

        net = [wn(cc.Conv1d(in_size, capacity, 15, padding=cc.get_padding(15)))]
        net.append(nn.LeakyReLU(0.2))

        for i in range(n_layers):
            net.append(
                wn(
                    cc.Conv1d(
                        capacity * multiplier**i,
                        min(1024, capacity * multiplier ** (i + 1)),
                        41,
                        stride=multiplier,
                        padding=cc.get_padding(41, multiplier),
                        groups=multiplier ** (i + 1),
                    )
                )
            )
            net.append(nn.LeakyReLU(0.2))

        net.append(
            wn(
                cc.Conv1d(
                    min(1024, capacity * multiplier ** (i + 1)),
                    min(1024, capacity * multiplier ** (i + 1)),
                    5,
                    padding=cc.get_padding(5),
                )
            )
        )
        net.append(nn.LeakyReLU(0.2))
        net.append(wn(cc.Conv1d(min(1024, capacity * multiplier ** (i + 1)), 1, 1)))
        self.net = nn.ModuleList(net)

    def forward(self, x):
        feature = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                feature.append(x)
        return feature


class StackDiscriminators(nn.Module):
    def __init__(self, n_dis, *args, **kwargs):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [Discriminator(*args, **kwargs) for i in range(n_dis)],
        )

    def forward(self, x):
        features = []
        for layer in self.discriminators:
            features.append(layer(x))
            x = nn.functional.avg_pool1d(x, 2)
        return features


class RAVE(nn.Module):
    def __init__(
        self,
        data_size,
        capacity,
        latent_size,
        ratios,
        bias,
        loud_stride,
        use_noise,
        noise_ratios,
        noise_bands,
        d_capacity,
        d_multiplier,
        d_n_layers,
        warmup,
        mode,
        no_latency=False,
        min_kl=1e-4,
        max_kl=5e-1,
        cropped_latent_size=0,
        feature_match=True,
        sr=24000,
    ):
        super().__init__()

        if data_size == 1:
            self.pqmf = None
        else:
            self.pqmf = PQMF(70 if no_latency else 100, data_size)

        self.loudness = Loudness(sr, 512)

        encoder_out_size = cropped_latent_size if cropped_latent_size else latent_size

        self.encoder = Encoder(
            data_size,
            capacity,
            encoder_out_size,
            ratios,
            "causal" if no_latency else "centered",
            bias,
        )
        self.decoder = Generator(
            latent_size,
            capacity,
            data_size,
            ratios,
            loud_stride,
            use_noise,
            noise_ratios,
            noise_bands,
            "causal" if no_latency else "centered",
            bias,
        )

        self.discriminator = StackDiscriminators(
            3,
            in_size=1,
            capacity=d_capacity,
            multiplier=d_multiplier,
            n_layers=d_n_layers,
        )

        self.idx = 0

        self.register_buffer("latent_pca", torch.eye(encoder_out_size))
        self.register_buffer("latent_mean", torch.zeros(encoder_out_size))
        self.register_buffer("fidelity", torch.zeros(encoder_out_size))

        self.latent_size = latent_size

        self.automatic_optimization = False

        self.warmup = warmup
        self.warmed_up = False
        self.sr = sr
        self.mode = mode

        self.min_kl = min_kl
        self.max_kl = max_kl
        self.cropped_latent_size = cropped_latent_size

        self.feature_match = feature_match

        self.register_buffer("saved_step", torch.tensor(0))

    def configure_optimizers(self):
        gen_p = list(self.encoder.parameters())
        gen_p += list(self.decoder.parameters())
        dis_p = list(self.discriminator.parameters())

        gen_opt = torch.optim.Adam(gen_p, 1e-4, (0.5, 0.9))
        dis_opt = torch.optim.Adam(dis_p, 1e-4, (0.5, 0.9))

        return gen_opt, dis_opt

    def lin_distance(self, x, y):
        return torch.norm(x - y) / torch.norm(x)

    def log_distance(self, x, y):
        return abs(torch.log(x + 1e-7) - torch.log(y + 1e-7)).mean()

    def distance(self, x, y):
        scales = [2048, 1024, 512, 256, 128]
        x = multiscale_stft(x, scales, 0.75)
        y = multiscale_stft(y, scales, 0.75)

        lin = sum(list(map(self.lin_distance, x, y)))
        log = sum(list(map(self.log_distance, x, y)))

        return lin + log

    def reparametrize(self, mean, scale):
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        if self.cropped_latent_size:
            noise = torch.randn(
                z.shape[0],
                self.latent_size - self.cropped_latent_size,
                z.shape[-1],
            ).to(z.device)
            z = torch.cat([z, noise], 1)
        return z, kl

    def adversarial_combine(self, score_real, score_fake, mode="hinge"):
        if mode == "hinge":
            loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
            loss_dis = loss_dis.mean()
            loss_gen = -score_fake.mean()
        elif mode == "square":
            loss_dis = (score_real - 1).pow(2) + score_fake.pow(2)
            loss_dis = loss_dis.mean()
            loss_gen = (score_fake - 1).pow(2).mean()
        elif mode == "nonsaturating":
            score_real = torch.clamp(torch.sigmoid(score_real), 1e-7, 1 - 1e-7)
            score_fake = torch.clamp(torch.sigmoid(score_fake), 1e-7, 1 - 1e-7)
            loss_dis = -(torch.log(score_real) + torch.log(1 - score_fake)).mean()
            loss_gen = -torch.log(score_fake).mean()
        else:
            raise NotImplementedError
        return loss_dis, loss_gen

    def encode(self, x):
        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x)
        z, _ = self.reparametrize(mean, scale)
        return z

    def decode(self, z):
        y = self.decoder(z, add_noise=True)
        if self.pqmf is not None:
            y = self.pqmf.inverse(y)
        return y

    def validation_step(self, batch, batch_idx):
        x = batch.unsqueeze(1)

        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x)
        z, _ = self.reparametrize(mean, scale)
        y = self.decoder(z, add_noise=self.warmed_up)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)
            y = self.pqmf.inverse(y)

        distance = self.distance(x, y)

        if self.trainer is not None:
            log.info("validation", distance)

        return torch.cat([x, y], -1), mean

    def validation_epoch_end(self, out):
        audio, z = list(zip(*out))

        if self.saved_step > self.warmup:
            self.warmed_up = True

        # LATENT SPACE ANALYSIS
        if not self.warmed_up:
            z = torch.cat(z, 0)
            z = rearrange(z, "b c t -> (b t) c")

            self.latent_mean.copy_(z.mean(0))
            z = z - self.latent_mean

            pca = PCA(z.shape[-1]).fit(z.cpu().numpy())

            components = pca.components_
            components = torch.from_numpy(components).to(z)
            self.latent_pca.copy_(components)

            var = pca.explained_variance_ / np.sum(pca.explained_variance_)
            var = np.cumsum(var)

            self.fidelity.copy_(torch.from_numpy(var).to(self.fidelity))

            var_percent = [0.8, 0.9, 0.95, 0.99]
            for p in var_percent:
                log.info(f"{p}%_manifold", np.argmax(var > p).astype(np.float32))

        y = torch.cat(audio, 0)[:64].reshape(-1)
        # self.logger.experiment.add_audio("audio_val", y, self.saved_step.item(), self.sr)
        self.idx += 1


def get_rave(name: str, hyper_cfg: HyperConfig) -> RAVE:
    cc.use_cached_conv(False)
    model = RAVE(
        data_size=16,
        capacity=32,
        latent_size=hyper_cfg.latent_dim,
        ratios=[4, 4, 2, 2, 2],
        bias=True,
        loud_stride=1,
        use_noise=False,
        noise_ratios=[],
        noise_bands=0,
        d_capacity=16,
        d_multiplier=4,
        d_n_layers=4,
        warmup=1000000,
        mode="hinge",
        no_latency=False,
        sr=16000,
        min_kl=1e-1,
        max_kl=1e-1,
        cropped_latent_size=0,
        feature_match=True,
    )
    return model

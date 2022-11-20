import torch
from torch import nn
from torch.nn import functional as F

from ..utils.cfg_classes import HyperConfig
from . import ae


class VAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reparam: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.reparam = reparam
        self.decoder = decoder

    def forward(self, data: torch.Tensor):
        e = self.encoder(data)
        z, mu, sigma = self.reparam(e)
        d = self.decoder(z)
        return d, mu, sigma


# TODO something prettier and less copypasty to deal with encoder returning a
# list (because of the possibility of multiple levels)
class ResVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reparam: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.reparam = reparam
        self.decoder = decoder

    def forward(self, data: torch.Tensor):
        e = self.encoder(data)
        [e] = e
        z, mu, sigma = self.reparam(e)
        d = self.decoder([z])
        return d, mu, sigma


class Reparametrization(nn.Module):
    def __init__(self, in_out_len: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_out_len = in_out_len
        self.fc_mu = nn.Linear(self.in_out_len * latent_dim, self.latent_dim)
        self.fc_sig = nn.Linear(self.in_out_len * latent_dim, self.latent_dim)
        self.fc_z = nn.Linear(self.latent_dim, self.in_out_len * latent_dim)

    def forward(self, data: torch.Tensor):
        flat_data = data.flatten(start_dim=1)
        mu = self.fc_mu(flat_data)
        log_sigma = self.fc_sig(flat_data)
        std = torch.exp(0.5 * log_sigma)
        # sample = torch.normal(mu, std)
        eps = torch.randn_like(std)
        sample = mu + eps * std
        fc_z_out = self.fc_z(sample)
        z = fc_z_out.view(-1, self.latent_dim, self.in_out_len)
        # z = self.fc_z(sample).view(-1, self.latent_dim, self.in_out_len)
        # raise Exception(f"Data shape: {data.shape}\nLatent_dim: {self.latent_dim}\nFlat data shape: {flat_data.shape}\nIn out len: {self.in_out_len}\nSample shape: {sample.shape}\nfc_z_out_shape: {fc_z_out.shape}\nz shape: {z.shape}\n{self}")
        return z, mu, std


def _create_vae(seq_length: int, latent_dim: int):
    encoder = ae.Encoder(seq_length, latent_dim)
    decoder = ae.Decoder(seq_length, latent_dim, encoder.output_lengths)
    reparam = Reparametrization(encoder.output_lengths[-1], latent_dim)
    return VAE(encoder, decoder, reparam)


def _create_res_vae(hyper_cfg: HyperConfig):
    encoder = ae.get_res_encoder(hyper_cfg)
    decoder = ae.get_res_decoder(hyper_cfg)
    assert hyper_cfg.res_ae.levels == 1, "Res-VAE with multiple levels not supported"
    encoder_out_seq_len = hyper_cfg.seq_len // (
        hyper_cfg.res_ae.strides_t[0] ** hyper_cfg.res_ae.downs_t[0]
    )
    reparam = Reparametrization(encoder_out_seq_len, hyper_cfg.latent_dim)
    return ResVAE(encoder, decoder, reparam)


def get_vae(name: str, hyper_cfg: HyperConfig):
    if name == "base":
        return _create_vae(hyper_cfg.seq_len, hyper_cfg.latent_dim)
    elif name == "res-vae":
        return _create_res_vae(hyper_cfg)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

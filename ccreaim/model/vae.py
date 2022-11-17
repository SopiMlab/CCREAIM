import torch
from torch import nn
from torch.nn import functional as F

from . import ae


class VAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reparam: nn.Module,
        kld_weight: float = 0.05,
    ):
        super().__init__()
        self.encoder = encoder
        self.reparam = reparam
        self.decoder = decoder
        self.kld_weight = kld_weight

    def forward(self, data: torch.Tensor):
        e = self.encoder(data)
        z, mu, sigma = self.reparam(e)
        d = self.decoder(z)
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
        sigma = torch.exp(log_sigma)
        sample = torch.normal(mu, sigma)
        z = self.fc_z(sample).view(-1, self.latent_dim, self.in_out_len)
        return z, mu, sigma


def _create_vae(seq_length: int, latent_dim: int):
    encoder = ae.Encoder(seq_length, latent_dim)
    decoder = ae.Decoder(seq_length, latent_dim, encoder.output_lengths)
    reparam = Reparametrization(encoder.output_lengths[-1], latent_dim)
    return VAE(encoder, decoder, reparam)


def get_vae(name: str, seq_length: int, latent_dim: int):
    if name == "base":
        return _create_vae(seq_length, latent_dim)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

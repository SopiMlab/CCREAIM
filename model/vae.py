import torch
from torch import nn
from torch.nn import functional as F

from model import ae


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, reparam: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.reparam = reparam
        self.decoder = decoder

    def forward(self, data: torch.Tensor):
        e = self.encoder(data)
        z, mu, sigma = self.reparam(e)
        d = self.decoder(z)
        return d, mu, sigma

    def loss_fn(
        self,
        data: torch.Tensor,
        pred: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        kld_weight: float = 1.0,
    ):
        mse = F.mse_loss(pred, data)
        kld = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return mse + kld_weight * kld


class Reparametrization(nn.Module):
    def __init__(self, enc_dec_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_dec_dim = enc_dec_dim
        self.fc_mu = nn.Linear(self.enc_dec_dim, self.latent_dim)
        self.fc_sig = nn.Linear(self.enc_dec_dim, self.latent_dim)
        self.fc_z = nn.Linear(self.latent_dim, self.enc_dec_dim)

    def forward(self, data: torch.Tensor):
        flat_data = data.flatten(start_dim=1)
        mu = self.fc_mu(flat_data)
        log_sigma = self.fc_sig(flat_data)
        sigma = torch.exp(log_sigma)
        samp = torch.normal(mu, sigma, mu.size())
        z = self.fc_z(samp)
        return z, mu, sigma


def _create_vae(seq_length: int):
    encoder = ae.Encoder(seq_length)
    decoder = ae.Decoder(seq_length, encoder.output_lengths)
    reparam = Reparametrization(encoder.output_lengths[-1] * 256, 256)
    return VAE(encoder, decoder, reparam)


def get_vae(name: str, seq_length: int):
    if name == "base":
        return _create_vae(seq_length)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

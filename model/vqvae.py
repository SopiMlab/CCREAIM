import torch
from torch import nn
from torch.nn import functional as F

from model import ae


class VQVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, vq: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.vq = vq
        self.decoder = decoder

    def forward(self, data: torch.Tensor):
        e = self.encoder(data)
        z = self.vq(e)
        # z, mu, sigma = self.reparam(e)
        d = self.decoder(z)
        return d  # , mu, sigma


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, data: torch.Tensor):

        return


def _create_vqvae(seq_length: int, latent_dim: int):
    encoder = ae.Encoder(seq_length, latent_dim)
    decoder = ae.Decoder(seq_length, latent_dim, encoder.output_lengths)
    reparam = VectorQuantizer(200, 256)
    return VQVAE(encoder, decoder, reparam)


def get_vqvae(name: str, seq_length: int, latent_dim: int):
    if name == "base":
        return _create_vqvae(seq_length, latent_dim)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

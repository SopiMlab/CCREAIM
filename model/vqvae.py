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

    def loss_fn(
        self,
        pred: torch.Tensor,
        data: torch.Tensor,
        beta: float = 1.0,
    ):
        mse = F.mse_loss(pred, data)
        return mse


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


def _create_vqvae(seq_length: int):
    encoder = ae.Encoder(seq_length)
    decoder = ae.Decoder(seq_length, encoder.output_lengths)
    reparam = VectorQuantizer()
    return VQVAE(encoder, decoder, reparam)


def get_vqvae(name: str, seq_length: int):
    if name == "base":
        return _create_vqvae(seq_length)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

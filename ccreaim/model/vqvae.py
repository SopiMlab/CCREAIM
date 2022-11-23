import logging

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.cfg_classes import HyperConfig
from . import ae

"""
Reference: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
"""

log = logging.getLogger(__name__)


class VQVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, vq: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.vq = vq
        self.decoder = decoder

    def forward(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.encoder(data)
        quantized_latents = self.vq(e)
        quantized_latents_res = e + (quantized_latents - e).detach()
        d = self.decoder(quantized_latents_res)
        return d, e, quantized_latents


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.transpose(1, 2).contiguous()  # [B x D x S] -> [B x S x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.cdist(latents, self.embedding.weight, p=2)  # [B x S x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=-1)  # [B x S]
        # Quantize the latents
        quantized_latents = self.embedding(encoding_inds)
        # [B x D x S]
        return quantized_latents.transpose(1, 2).contiguous()


def _create_vqvae(seq_length: int, latent_dim: int, num_embedings: int):
    encoder = ae.Encoder(seq_length, latent_dim)
    decoder = ae.Decoder(seq_length, latent_dim, encoder.output_lengths)
    reparam = VectorQuantizer(num_embedings, latent_dim)
    return VQVAE(encoder, decoder, reparam)


def get_vqvae(name: str, hyper_cfg: HyperConfig):
    if name == "base":
        return _create_vqvae(
            hyper_cfg.seq_len, hyper_cfg.latent_dim, hyper_cfg.vqvae.num_embeddings
        )
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

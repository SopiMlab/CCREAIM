import torch
from torch import nn

from ..utils.cfg_classes import HyperConfig
from . import ae, transformer, vae, vqvae


class E2E(nn.Module):
    def __init__(self, encoder: nn.Module, trf: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trf = trf

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(data)
        z = self.trf(enc)
        dec = self.encoder(z)
        return dec


def _create_e2e_ae(seq_length: int, num_seq: int, latent_dim: int):
    encoder = ae.Encoder(seq_length, latent_dim)
    decoder = ae.Decoder(seq_length, latent_dim, encoder.output_lengths)
    trf = transformer.Transformer(
        dim_model=num_seq,
        num_heads=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout_p=0.1,
    )
    return E2E(encoder, trf, decoder)


def _create_e2e_vae(seq_length: int, num_seq: int, latent_dim: int):
    encoder = ae.Encoder(seq_length, latent_dim)
    decoder = ae.Decoder(seq_length, latent_dim, encoder.output_lengths)
    trf = transformer.Transformer(
        dim_model=num_seq,
        num_heads=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout_p=0.1,
    )
    return E2E(encoder, trf, decoder)


def get_e2e(name: str, hyper_cfg: HyperConfig):
    if name == "base_ae":
        return _create_e2e_ae(
            hyper_cfg.seq_len, hyper_cfg.num_seq, hyper_cfg.latent_dim
        )
    elif name == "base_vae":
        return _create_e2e_vae(
            hyper_cfg.seq_len, hyper_cfg.num_seq, hyper_cfg.latent_dim
        )
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

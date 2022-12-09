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


def _create_e2e_ae(hyper_cfg: HyperConfig) -> E2E:
    encoder = ae.Encoder(hyper_cfg.seq_len, hyper_cfg.latent_dim)
    decoder = ae.Decoder(
        hyper_cfg.seq_len, hyper_cfg.latent_dim, encoder.output_lengths
    )
    trf = transformer.Transformer(
        dim_model=hyper_cfg.latent_dim,
        num_heads=hyper_cfg.latent_dim
        // hyper_cfg.transformer.num_heads_latent_dimension_div,
        num_encoder_layers=hyper_cfg.transformer.num_enc_layers,
        num_decoder_layers=hyper_cfg.transformer.num_dec_layers,
        dropout_p=0.1,
    )
    return E2E(encoder, trf, decoder)


def get_e2e(hyper_cfg: HyperConfig) -> E2E:
    if hyper_cfg.model == "e2e":
        return _create_e2e_ae(hyper_cfg)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(hyper_cfg.model))

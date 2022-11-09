import torch
from torch import nn

from model import ae, transformer, vae, vqvae


class EndToEnd(nn.Module):
    def __init__(
        self, encoder: nn.Module, trf: nn.Module, decoder: nn.Module, num_seq: int
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trf = trf

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        enc = torch.Tensor([self.encoder(seq) for seq in data.transpose(0, 1)])
        z_comb = self.trf(enc)
        dec = torch.Tensor([self.encoder(z) for z in z_comb.transpose(0, 1)])
        return dec


def _create_endtoend_ae(seq_length: int, num_seq: int, latent_dim: int):
    encoder = ae.Encoder(seq_length, latent_dim)
    decoder = ae.Decoder(seq_length, latent_dim, encoder.output_lengths)
    trf = transformer.Transformer(
        dim_model=num_seq,
        num_heads=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout_p=0.1,
    )
    return EndToEnd(encoder, trf, decoder, num_seq)


def get_end_to_end(name: str, seq_length: int, num_seq: int, latent_dim: int):
    if name == "base_ae":
        return _create_endtoend_ae(seq_length, num_seq, latent_dim)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

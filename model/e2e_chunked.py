import logging

import torch
from torch import nn

from model import ae, transformer, vae, vqvae

log = logging.getLogger(__name__)


class E2EChunked(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        trf: nn.Module,
        decoder: nn.Module,
        seq_length: int,
        latent_dim: int,
        seq_num: int,
        enc_out_length: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trf = trf
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.seq_num = seq_num
        self.enc_out_length = enc_out_length

    def forward(
        self, data: torch.Tensor, key_pad_mask: torch.Tensor, device  # TODO change this
    ) -> torch.Tensor:
        enc_batch = self.encoder(data.flatten(0, 1).unsqueeze(1))
        enc = enc_batch.view(-1, self.seq_num, self.latent_dim, self.enc_out_length)
        enc = enc.flatten(2, -1)
        enc_src = enc[:, :-1, :]
        enc_tgt = enc[:, 1:, :]
        tgt_mask = self.trf.get_tgt_mask(enc_tgt.size(1))
        tgt_mask = tgt_mask.to(device)
        z_comb = self.trf(
            enc_src,
            enc_tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=key_pad_mask[:, :-1],
            tgt_key_padding_mask=key_pad_mask[:, 1:],
        )
        z_comb = z_comb.view(-1, self.seq_num - 1, self.latent_dim, self.enc_out_length)
        z_comb = z_comb.flatten(0, 1)
        dec = self.decoder(z_comb)
        dec = dec.squeeze()
        dec = dec.view(-1, self.seq_num - 1, self.seq_length)
        return dec


def _create_e2e_chunked_ae(seq_length: int, seq_num: int, latent_dim: int):
    encoder = ae.Encoder(seq_length, latent_dim)
    decoder = ae.Decoder(seq_length, latent_dim, encoder.output_lengths)
    trf = transformer.Transformer(
        dim_model=latent_dim * encoder.output_lengths[-1],
        num_heads=latent_dim // 4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout_p=0.1,
    )
    return E2EChunked(
        encoder,
        trf,
        decoder,
        seq_length,
        latent_dim,
        seq_num,
        encoder.output_lengths[-1],
    )


def get_e2e_chunked(name: str, seq_length: int, num_seq: int, latent_dim: int):
    if name == "base_ae":
        return _create_e2e_chunked_ae(seq_length, num_seq, latent_dim)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

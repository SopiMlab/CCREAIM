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
        seq_cat: bool,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trf = trf
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.seq_num = seq_num
        self.enc_out_length = enc_out_length
        self.seq_cat = seq_cat
        self.shift_amount = self.enc_out_length if self.seq_cat else 1

    def forward(self, data: torch.Tensor, key_pad_mask: torch.Tensor) -> torch.Tensor:
        data = data.flatten(0, 1).unsqueeze(1)
        enc_batch = self.encoder(data).transpose(-1, -2)
        enc = enc_batch.view(-1, self.seq_num, self.enc_out_length, self.latent_dim)
        if self.seq_cat:
            enc = enc.flatten(1, 2)  # merge into sequence of vectors
            key_pad_mask = key_pad_mask.repeat_interleave(self.enc_out_length, dim=-1)
        else:
            enc = enc.flatten(2, -1)  # merge into one vector
        # shift by one or by self.enc_out_length
        enc_src = enc[:, : -self.shift_amount, :]
        enc_tgt = enc[:, self.shift_amount :, :]
        tgt_mask = self.trf.get_tgt_mask(enc_tgt.size(1))
        tgt_mask = tgt_mask.to(enc_tgt.device)
        z_comb = self.trf(
            enc_src,
            enc_tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=key_pad_mask[:, : -self.shift_amount],
            tgt_key_padding_mask=key_pad_mask[:, self.shift_amount :],
        )
        z_comb = z_comb.view(-1, self.seq_num - 1, self.enc_out_length, self.latent_dim)
        z_comb = z_comb.flatten(0, 1).transpose(-1, -2)
        dec = self.decoder(z_comb)
        dec = dec.squeeze()
        dec = dec.view(-1, self.seq_num - 1, self.seq_length)
        return dec


def _create_e2e_chunked_ae(
    seq_length: int, seq_num: int, latent_dim: int, seq_cat: bool
):
    encoder = ae.Encoder(seq_length, latent_dim)
    decoder = ae.Decoder(seq_length, latent_dim, encoder.output_lengths)
    trf = transformer.Transformer(
        dim_model=latent_dim if seq_cat else latent_dim * encoder.output_lengths[-1],
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
        seq_cat,
    )


def get_e2e_chunked(
    name: str, seq_length: int, num_seq: int, latent_dim: int, seq_cat: bool
):
    if name == "base_ae":
        return _create_e2e_chunked_ae(seq_length, num_seq, latent_dim, seq_cat)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

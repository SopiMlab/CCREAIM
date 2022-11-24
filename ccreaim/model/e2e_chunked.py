import logging

import torch
from torch import nn

from ..utils.cfg_classes import HyperConfig
from . import ae, transformer, vae, vqvae

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


def _create_e2e_chunked_ae(hyper_cfg: HyperConfig):
    seq_len = hyper_cfg.seq_len
    latent_dim = hyper_cfg.latent_dim
    encoder = ae.Encoder(seq_len, latent_dim)
    decoder = ae.Decoder(seq_len, latent_dim, encoder.output_lengths)
    trf = transformer.Transformer(
        dim_model=latent_dim
        if hyper_cfg.seq_cat
        else latent_dim * encoder.output_lengths[-1],
        num_heads=latent_dim // 4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout_p=0.1,
    )
    return E2EChunked(
        encoder,
        trf,
        decoder,
        seq_len,
        latent_dim,
        hyper_cfg.num_seq,
        encoder.output_lengths[-1],
        hyper_cfg.seq_cat,
    )


def _create_e2e_chunked_res_ae(hyper_cfg: HyperConfig):
    seq_len = hyper_cfg.seq_len
    latent_dim = hyper_cfg.latent_dim
    encoder = ae.get_res_encoder(hyper_cfg)
    decoder = ae.get_res_decoder(hyper_cfg)
    encoder_output_length = ae.res_encoder_output_seq_length(hyper_cfg)
    trf = transformer.Transformer(
        dim_model=latent_dim
        if hyper_cfg.seq_cat
        else latent_dim * encoder_output_length,
        num_heads=latent_dim // hyper_cfg.transformer.num_heads_latent_dimension_div,
        num_encoder_layers=hyper_cfg.transformer.num_enc_layers,
        num_decoder_layers=hyper_cfg.transformer.num_dec_layers,
        dropout_p=0.1,
    )
    return E2EChunked(
        encoder,
        trf,
        decoder,
        seq_len,
        latent_dim,
        hyper_cfg.num_seq,
        encoder_output_length,
        hyper_cfg.seq_cat,
    )


def get_e2e_chunked(name: str, hyper_cfg: HyperConfig):
    if name == "base_ae":
        return _create_e2e_chunked_ae(hyper_cfg)
    elif name == "base_res-ae":
        return _create_e2e_chunked_res_ae(hyper_cfg)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

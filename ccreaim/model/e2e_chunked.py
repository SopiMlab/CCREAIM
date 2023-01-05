from typing import Union

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import util
from ..utils.cfg_classes import HyperConfig
from . import ae, transformer, vae, vqvae


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
        self, data: torch.Tensor, key_pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = data.view(-1, 1, self.seq_length)
        enc_out_batch = self.encoder(data)
        enc_out_batch = enc_out_batch.transpose(-1, -2)
        enc_out = enc_out_batch.view(
            -1, self.seq_num, self.enc_out_length, self.latent_dim
        )
        enc_out_flat = enc_out.flatten(1, 2)  # merge into sequence of vectors

        # extend mask
        key_pad_mask = key_pad_mask.repeat_interleave(self.enc_out_length, dim=-1)
        src_key_pad_mask = key_pad_mask

        src = enc_out_flat
        # shift by one start token
        tgt = torch.cat(
            (
                torch.zeros_like(enc_out_flat[:, 0:1, :], device=enc_out_flat.device),
                enc_out_flat[:, :-1, :],
            ),
            dim=1,
        )
        tgt_key_pad_mask = torch.cat(
            (
                torch.zeros_like(key_pad_mask[:, 0:1], device=enc_out_flat.device),
                key_pad_mask[:, :-1],
            ),
            dim=1,
        )

        tgt_mask = self.trf.get_tgt_mask(tgt.size(1))
        tgt_mask = tgt_mask.to(tgt.device)
        trf_out_flat = self.trf(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_pad_mask,
            tgt_key_padding_mask=tgt_key_pad_mask,
        )
        trf_out = trf_out_flat.view(
            -1, self.seq_num, self.enc_out_length, self.latent_dim
        )
        trf_out_re_flat = trf_out.view(
            -1, self.enc_out_length, self.latent_dim
        ).transpose(-1, -2)

        dec_out = self.decoder(trf_out_re_flat)
        dec_out = dec_out.squeeze()
        dec_out = dec_out.view(-1, self.seq_num, self.seq_length)
        return dec_out, enc_out, trf_out

    def generate(
        self, data: torch.Tensor, in_chunks: int, feed_in_tokens: int, gen_chunks: int
    ) -> torch.Tensor:
        """Generate audio from an impot context

        Args:
            data (torch.Tensor): The input context to the generated audio
            in_chunks (int): The number of chunks in the input context
            feed_in_len (int): The number of 'tokens' to feed into the transformer decoder
            gen_chunks (int): The number of chunks generated

        Returns:
            torch.Tensor: The generated audio
        """
        data = data.view(-1, 1, self.seq_length)
        enc_out_batch = self.encoder(data)
        enc_out_batch = enc_out_batch.transpose(-1, -2)
        enc_out = enc_out_batch.view(
            -1, in_chunks, self.enc_out_length, self.latent_dim
        )
        enc_out_flat = enc_out.flatten(1, 2)  # merge into sequence of vectors

        src = enc_out_flat
        if feed_in_tokens == 0:
            tgt = torch.zeros_like(src[:, 0:1], device=src.device)
        else:
            tgt = src[:, 0:feed_in_tokens]

        for _ in range(gen_chunks * self.enc_out_length):
            tgt_mask = self.trf.get_tgt_mask(tgt.size(1))
            tgt_mask = tgt_mask.to(tgt.device)
            trf_pred = self.trf(src, tgt, tgt_mask=tgt_mask)
            tgt = torch.cat([tgt, trf_pred[:, -1:, :]], dim=1)
        trf_out_flat = tgt[:, feed_in_tokens:, :]
        trf_out = trf_out_flat.view(
            -1, gen_chunks, self.enc_out_length, self.latent_dim
        )
        trf_out_re_flat = trf_out.view(
            -1, self.enc_out_length, self.latent_dim
        ).transpose(-1, -2)

        dec_out = self.decoder(trf_out_re_flat)
        dec_out = dec_out.squeeze()
        dec_out = dec_out.view(-1, gen_chunks, self.seq_length)
        return dec_out


class E2EChunkedVQVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        vq: nn.Module,
        trf: nn.Module,
        decoder: nn.Module,
        seq_length: int,
        latent_dim: int,
        seq_num: int,
        enc_out_length: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.vq = vq
        self.decoder = decoder
        self.trf = trf
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.seq_num = seq_num
        self.enc_out_length = enc_out_length
        self.shift_amount = self.enc_out_length

    def forward(
        self, data: torch.Tensor, key_pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = data.view(-1, 1, self.seq_length)
        enc_out_batch = self.encoder(data)
        enc_out_batch = enc_out_batch.transpose(-1, -2)
        enc_out = enc_out_batch.view(
            -1, self.seq_num, self.enc_out_length, self.latent_dim
        )
        enc_out_flat = enc_out.flatten(1, 2)  # merge into sequence of vectors

        # VQ
        quantized_latents, vq_inds = self.vq(enc_out_flat.transpose(-1, -2))
        vq_inds = vq_inds.view(-1, self.seq_num, self.enc_out_length)

        quantized_latents = quantized_latents.transpose(-1, -2)
        quantized_latents_res = (
            enc_out_flat + (quantized_latents - enc_out_flat).detach()
        )
        vq_out = quantized_latents_res.view(
            -1, self.seq_num, self.enc_out_length, self.latent_dim
        )
        vq_out_flat = vq_out.flatten(1, 2)  # merge into sequence of vectors

        # Mask
        key_pad_mask = key_pad_mask.repeat_interleave(self.enc_out_length, dim=-1)
        src_key_pad_mask = key_pad_mask

        src = vq_out_flat
        # shift by one start token
        tgt = torch.cat(
            (
                torch.zeros_like(vq_out_flat[:, 0:1, :], device=vq_out_flat.device),
                vq_out_flat[:, :-1, :],
            ),
            dim=1,
        )
        tgt_key_pad_mask = torch.cat(
            (
                torch.zeros_like(key_pad_mask[:, 0:1], device=enc_out_flat.device),
                key_pad_mask[:, :-1],
            ),
            dim=1,
        )

        tgt_mask = self.trf.get_tgt_mask(tgt.size(1))
        tgt_mask = tgt_mask.to(tgt.device)
        trf_out_flat = self.trf(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_pad_mask,
            tgt_key_padding_mask=tgt_key_pad_mask,
        )
        trf_out = trf_out_flat.view(
            -1, self.seq_num, self.enc_out_length, self.vq.K  # num_embeddings
        )

        # VQ lookup
        emb_ids = trf_out.argmax(-1)
        emb_ids_flat = emb_ids.view(-1, self.enc_out_length)
        trf_vq_out = self.vq.lookup(emb_ids_flat)

        trf_vq_out = trf_vq_out.transpose(-1, -2)
        dec_out = self.decoder(trf_vq_out)
        dec_out = dec_out.squeeze()
        dec_out = dec_out.view(-1, self.seq_num, self.seq_length)
        return (
            dec_out,
            enc_out,
            vq_out,
            vq_inds,
            trf_out,
        )

    def generate(
        self, data: torch.Tensor, in_chunks: int, feed_in_tokens: int, gen_chunks: int
    ) -> torch.Tensor:
        """Generate audio from an impot context

        Args:
            data (torch.Tensor): The input context to the generated audio
            in_chunks (int): The number of chunks in the input context
            feed_in_len (int): The number of 'tokens' to feed into the transformer decoder
            gen_chunks (int): The number of chunks generated

        Returns:
            torch.Tensor: The generated audio
        """
        data = data.view(-1, 1, self.seq_length)
        enc_out_batch = self.encoder(data)
        enc_out_batch = enc_out_batch.transpose(-1, -2)
        enc_out = enc_out_batch.view(
            -1, in_chunks, self.enc_out_length, self.latent_dim
        )
        enc_out_flat = enc_out.flatten(1, 2)  # merge into sequence of vectors
        # VQ
        quantized_enc, _ = self.vq(enc_out_flat.transpose(-1, -2))
        quantized_enc = quantized_enc.transpose(-1, -2)

        src = quantized_enc
        if feed_in_tokens == 0:
            tgt = torch.zeros_like(src[:, 0:1], device=src.device)
        else:
            tgt = src[:, 0:feed_in_tokens]

        for _ in range(gen_chunks * self.enc_out_length):
            tgt_mask = self.trf.get_tgt_mask(tgt.size(1))
            tgt_mask = tgt_mask.to(tgt.device)
            trf_out_flat = self.trf(src, tgt, tgt_mask=tgt_mask)
            trf_pred = trf_out_flat[:, -1:, :]
            # VQ lookup
            emb_ids = trf_pred.argmax(-1)
            emb_ids_flat = emb_ids.flatten(0, 1)
            trf_vq_out = self.vq.lookup(emb_ids_flat)
            tgt = torch.cat([tgt, trf_vq_out.unsqueeze(0)], dim=1)

        trf_out_flat = tgt[:, feed_in_tokens:, :]
        trf_out = trf_out_flat.view(
            -1, gen_chunks, self.enc_out_length, self.latent_dim
        )
        trf_out_re_flat = trf_out.view(
            -1, self.enc_out_length, self.latent_dim
        ).transpose(-1, -2)

        dec_out = self.decoder(trf_out_re_flat)
        dec_out = dec_out.squeeze()
        dec_out = dec_out.view(-1, gen_chunks, self.seq_length)
        return dec_out


def prepare_data_for_transformer(
    data, encoder, vq, seq_len, num_seq, enc_out_length, latent_dim
):
    data_orig = data
    data = data.view(-1, 1, seq_len)
    enc_out_batch = encoder(data)
    enc_out_batch = enc_out_batch.transpose(-1, -2)
    enc_out = enc_out_batch.view(-1, num_seq, enc_out_length, latent_dim)
    enc_out_flat = enc_out.flatten(1, 2)  # merge into sequence of vectors

    # VQ
    quantized_latents, vq_inds = vq(enc_out_flat.transpose(-1, -2))
    vq_inds = vq_inds.view(-1, num_seq, enc_out_length)

    quantized_latents = quantized_latents.transpose(-1, -2)
    quantized_latents_res = enc_out_flat + (quantized_latents - enc_out_flat).detach()
    vq_out = quantized_latents_res.view(-1, num_seq, enc_out_length, latent_dim)
    vq_out_flat = vq_out.flatten(1, 2)  # merge into sequence of vectors

    return vq_out_flat, vq_inds


def _create_e2e_chunked_ae(hyper_cfg: HyperConfig) -> E2EChunked:
    seq_len = hyper_cfg.seq_len
    latent_dim = hyper_cfg.latent_dim
    encoder = ae.Encoder(seq_len, latent_dim)
    decoder = ae.Decoder(seq_len, latent_dim, encoder.output_lengths)
    trf = transformer.Transformer(
        dim_model=latent_dim
        if hyper_cfg.seq_cat
        else latent_dim * encoder.output_lengths[-1],
        num_heads=latent_dim // hyper_cfg.transformer.num_heads_latent_dimension_div,
        num_encoder_layers=hyper_cfg.transformer.num_enc_layers,
        num_decoder_layers=hyper_cfg.transformer.num_dec_layers,
        dropout_p=0.1,
        linear_map=False,
    )
    return E2EChunked(
        encoder,
        trf,
        decoder,
        seq_len,
        latent_dim,
        hyper_cfg.num_seq,
        encoder.output_lengths[-1],
    )


def _create_e2e_chunked_res_ae(hyper_cfg: HyperConfig) -> E2EChunked:
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
        linear_map=False,
    )

    if hyper_cfg.pre_trained_ae_path is not None:
        encoder, decoder = util.load_pre_trained_ae(hyper_cfg, encoder, decoder)
    if hyper_cfg.pre_trained_transformer_path is not None:
        trf = util.load_pre_trained_transformer(hyper_cfg, trf)

    return E2EChunked(
        encoder,
        trf,
        decoder,
        seq_len,
        latent_dim,
        hyper_cfg.num_seq,
        encoder_output_length,
    )


def _create_e2e_chunked_res_vqvae(hyper_cfg: HyperConfig) -> E2EChunkedVQVAE:
    seq_len = hyper_cfg.seq_len
    latent_dim = hyper_cfg.latent_dim
    encoder = ae.get_res_encoder(hyper_cfg)
    decoder = ae.get_res_decoder(hyper_cfg)
    encoder_output_length = ae.res_encoder_output_seq_length(hyper_cfg)
    trf = transformer.Transformer(
        dim_model=latent_dim,
        num_heads=latent_dim // hyper_cfg.transformer.num_heads_latent_dimension_div,
        num_encoder_layers=hyper_cfg.transformer.num_enc_layers,
        num_decoder_layers=hyper_cfg.transformer.num_dec_layers,
        dropout_p=0.1,
        linear_map=hyper_cfg.transformer.linear_map,
        num_embeddings=hyper_cfg.vqvae.num_embeddings,
    )
    vq = vqvae.VectorQuantizer(
        hyper_cfg.vqvae.num_embeddings,
        hyper_cfg.latent_dim,
        hyper_cfg.vqvae.reset_patience,
    )

    if hyper_cfg.pre_trained_vqvae_path is not None:
        encoder, vq, decoder = util.load_pre_trained_vqvae(
            hyper_cfg, encoder, vq, decoder
        )

    if hyper_cfg.pre_trained_transformer_path is not None:
        trf = util.load_pre_trained_transformer(hyper_cfg, trf)

    return E2EChunkedVQVAE(
        encoder,
        vq,
        trf,
        decoder,
        seq_len,
        latent_dim,
        hyper_cfg.num_seq,
        encoder_output_length,
    )


def get_e2e_chunked(hyper_cfg: HyperConfig) -> Union[E2EChunked, E2EChunkedVQVAE]:
    if hyper_cfg.model == "e2e-chunked":
        return _create_e2e_chunked_ae(hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked_res-ae":
        return _create_e2e_chunked_res_ae(hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked_res-vqvae":
        return _create_e2e_chunked_res_vqvae(hyper_cfg)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(hyper_cfg.model))

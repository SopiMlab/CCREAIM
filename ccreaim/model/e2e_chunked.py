import itertools
import logging
from typing import Union

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

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

    def generate(
        self, data: torch.Tensor, in_len: int, feed_in_len: int, gen_len: int
    ) -> torch.Tensor:
        data = data.flatten(0, 1).unsqueeze(1)
        enc_batch = self.encoder(data).transpose(-1, -2)
        enc = enc_batch.view(-1, in_len, self.enc_out_length, self.latent_dim)
        if self.seq_cat:
            enc = enc.flatten(1, 2)  # merge into sequence of vectors
        else:
            enc = enc.flatten(2, -1)

        enc_src = enc
        if feed_in_len == 0:
            tgt = torch.zeros_like(enc_src[:, 0:1], device=enc_src.device)
        else:
            tgt = enc_src[:, 0 : feed_in_len * self.enc_out_length]

        for i in range(gen_len * self.enc_out_length):
            tgt_mask = self.trf.get_tgt_mask(tgt.size(1))
            tgt_mask = tgt_mask.to(tgt.device)
            trf_pred = self.trf(enc_src, tgt, tgt_mask=tgt_mask)
            tgt = torch.cat([tgt, trf_pred[:, -1:, :]], dim=1)

        z_comb = tgt[:, feed_in_len:, :].view(
            -1, gen_len, self.enc_out_length, self.latent_dim
        )
        # z_comb = tgt.view(-1, gen_len + feed_in_len, self.enc_out_length, self.latent_dim)
        z_comb = z_comb.flatten(0, 1).transpose(-1, -2)
        dec = self.decoder(z_comb)
        dec = dec.squeeze()
        dec = dec.view(-1, gen_len + feed_in_len, self.seq_length)

        return dec


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
        seq_cat: bool,
        num_tokens: int,
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
        self.seq_cat = seq_cat
        self.shift_amount = self.enc_out_length if self.seq_cat else 1
        self.trf_out_to_tokens = nn.Linear(latent_dim, num_tokens)

    def forward(
        self, data: torch.Tensor, key_pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = data.flatten(0, 1).unsqueeze(1)
        enc_batch = (
            self.encoder(data)
            .transpose(-1, -2)
            .view(-1, self.seq_num, self.enc_out_length, self.latent_dim)
        )
        # VQ
        enc_flat = enc_batch.flatten(1, 2)
        quantized_latents = self.vq(enc_flat.transpose(-1, -2)).transpose(-1, -2)
        quantized_latents_res = enc_flat + (quantized_latents - enc_flat).detach()
        enc = quantized_latents_res.view(
            -1, self.seq_num, self.enc_out_length, self.latent_dim
        )
        if self.seq_cat:
            enc = enc.flatten(1, 2)  # merge into sequence of vectors
            key_pad_mask = key_pad_mask.repeat_interleave(self.enc_out_length, dim=-1)
        else:
            enc = enc.flatten(2, -1)  # merge into one vector
        # shift by one or by self.enc_out_length
        enc_src = quantized_latents_res[:, : -self.shift_amount, :]
        enc_tgt = quantized_latents_res[:, self.shift_amount :, :]
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
        # Transform to probability over tokens
        z_soft = F.softmax(self.trf_out_to_tokens(z_comb))

        # VQ lookup
        z_ids = z_soft.argmax(-1)
        z_quant = self.vq.lookup(z_ids.flatten(0, 1))

        z_quant = z_quant.transpose(-1, -2)
        dec = self.decoder(z_quant)
        dec = dec.squeeze()
        dec = dec.view(-1, self.seq_num - 1, self.seq_length)
        return (
            dec,
            enc,
            quantized_latents,
        )

    def generate(
        self, data: torch.Tensor, in_len: int, feed_in_len: int, gen_len: int
    ) -> torch.Tensor:
        data = data.flatten(0, 1).unsqueeze(1)
        enc_batch = self.encoder(data).transpose(-1, -2)
        enc = enc_batch.view(-1, in_len, self.enc_out_length, self.latent_dim)
        if self.seq_cat:
            enc = enc.flatten(1, 2)  # merge into sequence of vectors
        else:
            enc = enc.flatten(2, -1)

        enc_src = enc
        if feed_in_len == 0:
            tgt = torch.zeros_like(enc_src[:, 0:1], device=enc_src.device)
        else:
            tgt = enc_src[:, 0 : feed_in_len * self.enc_out_length]

        for i in range(gen_len * self.enc_out_length):
            tgt_mask = self.trf.get_tgt_mask(tgt.size(1))
            tgt_mask = tgt_mask.to(tgt.device)
            trf_pred = self.trf(enc_src, tgt, tgt_mask=tgt_mask)
            tgt = torch.cat([tgt, trf_pred[:, -1:, :]], dim=1)

        z_comb = tgt[:, feed_in_len:, :].view(
            -1, gen_len, self.enc_out_length, self.latent_dim
        )
        # z_comb = tgt.view(-1, gen_len + feed_in_len, self.enc_out_length, self.latent_dim)
        z_comb = z_comb.flatten(0, 1).transpose(-1, -2)
        dec = self.decoder(z_comb)
        dec = dec.squeeze()
        dec = dec.view(-1, gen_len + feed_in_len, self.seq_length)

        return dec


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
    )
    vq = vqvae.VectorQuantizer(
        hyper_cfg.vqvae.num_embeddings,
        hyper_cfg.latent_dim,
        hyper_cfg.vqvae.reset_patience,
    )

    if hyper_cfg.pre_trained_model_path is not None:
        checkpoint = torch.load(hyper_cfg.pre_trained_model_path, map_location="cpu")
        pretrained_state_dict = checkpoint["model_state_dict"]
        hyper_cfg_schema = OmegaConf.structured(HyperConfig)
        conf = OmegaConf.create(checkpoint["hyper_config"])
        pretrained_hyper_cfg = OmegaConf.merge(hyper_cfg_schema, conf)

        if (
            hyper_cfg.latent_dim == pretrained_hyper_cfg.latent_dim
            and hyper_cfg.seq_len == pretrained_hyper_cfg.seq_len
            and hyper_cfg.res_ae.downs_t == pretrained_hyper_cfg.res_ae.downs_t
            and hyper_cfg.res_ae.strides_t == pretrained_hyper_cfg.res_ae.strides_t
            and hyper_cfg.res_ae.input_emb_width
            == pretrained_hyper_cfg.res_ae.input_emb_width
            and hyper_cfg.res_ae.block_width == pretrained_hyper_cfg.res_ae.block_width
            and hyper_cfg.res_ae.block_depth == pretrained_hyper_cfg.res_ae.block_depth
            and hyper_cfg.res_ae.block_m_conv
            == pretrained_hyper_cfg.res_ae.block_m_conv
            and hyper_cfg.res_ae.block_dilation_growth_rate
            == pretrained_hyper_cfg.res_ae.block_dilation_growth_rate
            and hyper_cfg.res_ae.block_dilation_cycle
            == pretrained_hyper_cfg.res_ae.block_dilation_cycle
            and hyper_cfg.vqvae.num_embeddings
            == pretrained_hyper_cfg.vqvae.num_embeddings
        ):
            tmp_vq = vqvae.VQVAE(encoder, decoder, vq)
            tmp_vq.load_state_dict(pretrained_state_dict)
            encoder = tmp_vq.encoder
            vq = tmp_vq.vq
            decoder = tmp_vq.decoder
            if hyper_cfg.freeze_pre_trained:
                encoder.requires_grad_(False)
                # vq is frozen in operate by emedding.grad = 0
                decoder.requires_grad_(False)
        else:
            raise ValueError(
                f"Pre-trained config is not matching current config:\n"
                "\t\t\t\tCurrent config\t---\tPre-trained config\n"
                "latent_dim:\t\t\t\t"
                f"{hyper_cfg.latent_dim}"
                "\t---\t"
                f"{pretrained_hyper_cfg.latent_dim}\n"
                "seq_len:\t\t\t\t"
                f"{hyper_cfg.seq_len}"
                "\t---\t"
                f"{pretrained_hyper_cfg.seq_len}\n"
                "res_ae.downs_t:\t\t\t\t"
                f"{hyper_cfg.res_ae.downs_t}"
                "\t---\t"
                f"{pretrained_hyper_cfg.res_ae.downs_t}\n"
                "res_ae.strides_t:\t\t\t"
                f"{hyper_cfg.res_ae.strides_t}"
                "\t---\t"
                f"{pretrained_hyper_cfg.res_ae.strides_t}\n"
                "res_ae.input_emb_width:\t\t\t"
                f"{hyper_cfg.res_ae.input_emb_width}"
                "\t---\t"
                f"{pretrained_hyper_cfg.res_ae.input_emb_width}\n"
                "res_ae.block_width:\t\t\t"
                f"{hyper_cfg.res_ae.block_width}"
                "\t---\t"
                f"{pretrained_hyper_cfg.res_ae.block_width}\n"
                "res_ae.block_depth:\t\t\t"
                f"{hyper_cfg.res_ae.block_depth}"
                "\t---\t"
                f"{pretrained_hyper_cfg.res_ae.block_depth}\n"
                "res_ae.block_m_conv:\t\t\t"
                f"{hyper_cfg.res_ae.block_m_conv}"
                "\t---\t"
                f"{pretrained_hyper_cfg.res_ae.block_m_conv}\n"
                "res_ae.block_dilation_growth_rate:\t"
                f"{hyper_cfg.res_ae.block_dilation_growth_rate}"
                "\t---\t"
                f"{pretrained_hyper_cfg.res_ae.block_dilation_growth_rate}\n"
                "res_ae.block_dilation_cycle:\t\t"
                f"{hyper_cfg.res_ae.block_dilation_cycle}"
                "\t---\t"
                f"{pretrained_hyper_cfg.res_ae.block_dilation_cycle}\n"
                "vqvae.num_embeddings:\t\t\t"
                f"{hyper_cfg.vqvae.num_embeddings}"
                "\t---\t"
                f"{pretrained_hyper_cfg.vqvae.num_embeddings}\n"
            )

    return E2EChunkedVQVAE(
        encoder,
        vq,
        trf,
        decoder,
        seq_len,
        latent_dim,
        hyper_cfg.num_seq,
        encoder_output_length,
        hyper_cfg.seq_cat,
        hyper_cfg.vqvae.num_embeddings,
    )


def get_e2e_chunked(hyper_cfg: HyperConfig) -> Union[E2EChunked, E2EChunkedVQVAE]:
    if hyper_cfg.model == "e2e-chunked":
        return _create_e2e_chunked_ae(hyper_cfg)
    elif hyper_cfg.model == "be2e-chunked_res-ae":
        return _create_e2e_chunked_res_ae(hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked_res-vqvae":
        return _create_e2e_chunked_res_vqvae(hyper_cfg)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(hyper_cfg.model))

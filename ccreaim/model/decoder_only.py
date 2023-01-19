import logging
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.cfg_classes import HyperConfig
from .transformer import PositionalEncoding


class CachedDecoderOnly(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_layers: int,
        dropout_p: float,
        linear_map: bool = False,
        num_embeddings: int = 0,
    ):
        super().__init__()

        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=10000
        )

        # Using the Encoder class as the transformer decoder,
        # since it does not require the memory (from the encoder) as input in the forward pass
        self.transformer_decoder = CachedTransformerEncoder(
            encoder_layer=CachedTransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dropout=dropout_p,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        if linear_map:
            self.trf_out_to_tokens = nn.Linear(dim_model, num_embeddings)
        else:
            self.trf_out_to_tokens = nn.Identity()

    def forward(
        self,
        tgt: torch.Tensor,
        *,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        no_cache: bool = False,
    ) -> torch.Tensor:
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (batch_size, sequence length, num_tokens)
        trf_out = self.transformer_decoder(
            src=tgt,
            src_mask=tgt_mask,
            src_key_padding_mask=tgt_key_padding_mask,
            no_cache=no_cache,
        )
        if not self.training and not no_cache:
            trf_out = trf_out[0]

        out = self.trf_out_to_tokens(trf_out)
        return out

    def generate(self, tgt: torch.Tensor, gen_tokens: int) -> None:
        cache = None
        for i in range(gen_tokens):
            tgt_chunk = tgt[:, : gen_tokens + i, :]

            tgt_chunk = self.positional_encoder(tgt_chunk)
            trf_out_flat, cache = self.transformer_decoder(tgt_chunk, cache=cache)
            trf_out = self.trf_out_to_tokens(trf_out_flat)
            trf_pred = trf_out[:, -1:, :]

            ids = trf_pred.argmax(-1)
            ids_one_hot = F.one_hot(ids.long(), num_classes=256).int()

            tgt[:, gen_tokens + i] = ids_one_hot.squeeze()

    def generate_chunks(
        self,
        audio: torch.Tensor,
        tgt_chunk: int,
    ) -> torch.Tensor:
        num_chunks = audio.size(1) // tgt_chunk
        gen = torch.zeros(
            (audio.size(0), (num_chunks + 1) * tgt_chunk, 256), device=audio.device
        )
        gen[:, :tgt_chunk, :] = audio[:, :tgt_chunk, :]
        for chunk in range(num_chunks):
            tgt = gen[:, chunk * tgt_chunk : chunk * tgt_chunk + 2 * tgt_chunk]
            self.generate(tgt, tgt_chunk)
        return gen


class CachedTransformerEncoder(nn.TransformerEncoder):
    """Implementation taken from https://scale.com/blog/pytorch-improvements and
    https://github.com/alex-matton/causal-transformer-decoder/blob/master/causal_transformer_decoder/model.py
    """

    def forward(
        self,
        src: torch.Tensor,
        *,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        no_cache: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        output = src

        if self.training or no_cache:
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    no_cache=no_cache,
                )

            return output

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(
                output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=1)
        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=2)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class CachedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(
        self,
        src: torch.Tensor,
        *,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        no_cache: bool = False,
    ) -> torch.Tensor:

        if self.training or no_cache:
            return super().forward(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        src_last_tok = src[:, -1:, :]
        # self attention
        tmp_tgt = self.self_attn(
            src_last_tok,
            src,
            src,
            attn_mask=None,  # not needed because we only care about the last token
            key_padding_mask=src_key_padding_mask,
        )[0]
        src_last_tok = src_last_tok + self.dropout1(tmp_tgt)
        src_last_tok = self.norm1(src_last_tok)
        # final feed-forward network
        src_last_tok = self.norm2(src_last_tok + self._ff_block(src_last_tok))

        return src_last_tok


def get_decoder(hyper_cfg: HyperConfig) -> CachedDecoderOnly:
    if hyper_cfg.model == "transformer-decoder-only":
        return CachedDecoderOnly(
            dim_model=hyper_cfg.latent_dim,
            num_heads=hyper_cfg.latent_dim
            // hyper_cfg.transformer.num_heads_latent_dimension_div,
            num_layers=hyper_cfg.transformer.num_dec_layers,
            dropout_p=0.1,
            linear_map=hyper_cfg.transformer.linear_map,
            num_embeddings=hyper_cfg.vqvae.num_embeddings,
        )
    else:
        raise ValueError(f"Transformer model not implemented: {hyper_cfg.model}")

import logging
import math
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.cfg_classes import HyperConfig


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, dropout_p: float = 0.1, max_len: int = 1000):
        super().__init__()
        # Taken from https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Residual connection + pos encoding
        return self.dropout(x + self.pos_encoding[:, : x.size(1), :])  # type: ignore


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """

    # Constructor
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout_p: float,
        linear_map: bool = False,
        num_embeddings: int = 0,
    ):
        super().__init__()

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=10000
        )
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True,
        )

        if linear_map:
            self.trf_out_to_tokens = nn.Linear(dim_model, num_embeddings)
        else:
            self.trf_out_to_tokens = nn.Identity()

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        *,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (batch_size, sequence length, num_tokens)
        trf_out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_mask=src_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        out = self.trf_out_to_tokens(trf_out)
        return out

    def get_tgt_mask(self, size: int) -> torch.Tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        return mask

    def generate(
        self, src: torch.Tensor, tgt: torch.Tensor, gen_tokens: int, first: bool
    ) -> None:

        for i in range(gen_tokens):
            if not first:
                tgt_chunk = tgt[:, : gen_tokens + i, :]
            else:
                tgt_chunk = tgt[:, : 1 + i, :]

            trf_out_flat = self.forward(src, tgt_chunk)
            trf_pred = trf_out_flat[:, -1:, :]
            ids = trf_pred.argmax(-1)
            ids_one_hot = F.one_hot(ids.long(), num_classes=256).int()

            if not first:
                tgt[:, gen_tokens + i, :] = ids_one_hot.squeeze()
            else:
                tgt[:, 1 + i, :] = ids_one_hot.squeeze()

    def generate_chunks(
        self,
        audio: torch.Tensor,
        src_chunk: int,
        src_shift: int,
        tgt_chunk: int,
    ) -> torch.Tensor:
        num_chunks = audio.size(1) // src_shift
        gen = torch.zeros(
            (audio.size(0), num_chunks * tgt_chunk + 1, 256), device=audio.device
        )

        src = audio[:, :src_chunk, :]
        tgt = gen[:, : tgt_chunk + 1, :]
        self.generate(src, tgt, tgt_chunk, True)

        for chunk in range(num_chunks - 1):
            src = audio[
                :,
                (chunk + 1) * src_shift : (chunk + 1) * src_shift + src_chunk,
            ]
            tgt = gen[:, chunk * tgt_chunk : chunk * tgt_chunk + 2 * tgt_chunk]
            self.generate(src, tgt, tgt_chunk, False)

        gen = gen[:, 1:, :]
        return gen


class CachedTransformerDecoder(nn.TransformerDecoder):
    """Implementation taken from https://scale.com/blog/pytorch-improvements and
    https://github.com/alex-matton/causal-transformer-decoder/blob/master/causal_transformer_decoder/model.py
    """

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
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

        output = tgt

        if self.training:
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

            return output

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=1)
        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=2)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class CachedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.training:
            return super().forward(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        tgt_last_tok = tgt[:, -1:, :]
        # self attention
        tmp_tgt = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # not needed because we only care about the last token
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)
        # encoder-decoder attention
        tmp_tgt = self.multihead_attn(
            tgt_last_tok,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
        tgt_last_tok = self.norm2(tgt_last_tok)
        # final feed-forward network
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )

        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok


class CachedTransformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """

    # Constructor
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout_p: float,
        linear_map: bool = False,
        num_embeddings: int = 0,
    ):
        super().__init__()

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=10000
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dropout=dropout_p,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        self.transformer_decoder = CachedTransformerDecoder(
            decoder_layer=CachedTransformerDecoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dropout=dropout_p,
                batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )

        if linear_map:
            self.trf_out_to_tokens = nn.Linear(dim_model, num_embeddings)
        else:
            self.trf_out_to_tokens = nn.Identity()

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        *,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (batch_size, sequence length, num_tokens)
        memory = self.transformer_encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        trf_dec_out = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        if not self.training:
            trf_dec_out = trf_dec_out[0]

        out = self.trf_out_to_tokens(trf_dec_out)
        return out

    def get_tgt_mask(self, size: int) -> torch.Tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        return mask

    def generate(
        self, src: torch.Tensor, tgt: torch.Tensor, gen_tokens: int, first: bool
    ) -> None:
        memory = self.transformer_encoder(src)

        cache = None
        for i in range(gen_tokens):
            if not first:
                tgt_chunk = tgt[:, : gen_tokens + i, :]
            else:
                tgt_chunk = tgt[:, : 1 + i, :]
            trf_out_flat, cache = self.transformer_decoder(
                memory, tgt_chunk, cache=cache
            )
            trf_pred = trf_out_flat[:, -1:, :]

            ids = trf_pred.argmax(-1)
            ids_one_hot = F.one_hot(ids.long(), num_classes=256).int()

            if not first:
                tgt[:, gen_tokens + i, :] = ids_one_hot.squeeze()
            else:
                tgt[:, 1 + i, :] = ids_one_hot.squeeze()

    def generate_chunks(
        self,
        audio: torch.Tensor,
        src_chunk: int,
        src_shift: int,
        tgt_chunk: int,
    ) -> torch.Tensor:
        num_chunks = audio.size(1) // src_shift
        gen = torch.zeros(
            (audio.size(0), num_chunks * tgt_chunk + 1, 256), device=audio.device
        )

        src = audio[:, :src_chunk, :]
        tgt = gen[:, : tgt_chunk + 1, :]
        self.generate(src, tgt, tgt_chunk, True)

        for chunk in range(num_chunks - 1):
            src = audio[
                :,
                (chunk + 1) * src_shift : (chunk + 1) * src_shift + src_chunk,
            ]
            tgt = gen[:, chunk * tgt_chunk : chunk * tgt_chunk + 2 * tgt_chunk]
            self.generate(src, tgt, tgt_chunk, False)

        gen = gen[:, 1:, :]
        return gen


def get_transformer(hyper_cfg: HyperConfig) -> Union[Transformer, CachedTransformer]:
    if hyper_cfg.model == "transformer":
        return Transformer(
            dim_model=hyper_cfg.latent_dim,
            num_heads=hyper_cfg.latent_dim
            // hyper_cfg.transformer.num_heads_latent_dimension_div,
            num_encoder_layers=hyper_cfg.transformer.num_enc_layers,
            num_decoder_layers=hyper_cfg.transformer.num_dec_layers,
            dropout_p=0.1,
            linear_map=hyper_cfg.transformer.linear_map,
            num_embeddings=hyper_cfg.vqvae.num_embeddings,
        )
    elif hyper_cfg.model == "cached-transformer":
        return CachedTransformer(
            dim_model=hyper_cfg.latent_dim,
            num_heads=hyper_cfg.latent_dim
            // hyper_cfg.transformer.num_heads_latent_dimension_div,
            num_encoder_layers=hyper_cfg.transformer.num_enc_layers,
            num_decoder_layers=hyper_cfg.transformer.num_dec_layers,
            dropout_p=0.1,
            linear_map=hyper_cfg.transformer.linear_map,
            num_embeddings=hyper_cfg.vqvae.num_embeddings,
        )
    else:
        raise ValueError(f"Transformer model not implemented: {hyper_cfg.model}")

from torch import nn

from ..utils import util
from ..utils.cfg_classes import HyperConfig, ResAeConfig
from .resnet import Resnet1D


# Jukebox imitating ResNet-based AE:
def assert_shape(x, exp_shape):
    # This could move to util/be removed from productions version
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"


class EncoderConvBlock(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        res_scale=False,
    ):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    nn.Conv1d(
                        input_emb_width if i == 0 else width,
                        width,
                        filter_t,
                        stride_t,
                        pad_t,
                    ),
                    Resnet1D(
                        width,
                        depth,
                        m_conv,
                        dilation_growth_rate,
                        dilation_cycle,
                        res_scale,
                    ),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DecoderConvBlock(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        res_scale=False,
    ):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(
                        width,
                        depth,
                        m_conv,
                        dilation_growth_rate,
                        dilation_cycle,
                        res_scale=res_scale,
                    ),
                    nn.ConvTranspose1d(
                        width,
                        input_emb_width if i == (down_t - 1) else width,
                        filter_t,
                        stride_t,
                        pad_t,
                    ),
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# Multilevel AE:s from jukebox implementation
class MultiLevelResEncoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        levels,
        downs_t,
        strides_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(
            input_emb_width if level == 0 else output_emb_width,
            output_emb_width,
            down_t,
            stride_t,
            **block_kwargs_copy,
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t**down_t)
            # ssert_shape(x, (N, emb, T))
            xs.append(x)

        return xs


class MultiLevelResDecoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        levels,
        downs_t,
        strides_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels

        self.downs_t = downs_t

        self.strides_t = strides_t

        level_block = lambda level, down_t, stride_t: DecoderConvBlock(
            output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(
            list(zip(list(range(self.levels)), self.downs_t, self.strides_t))
        )
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t**down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x


# Single level AE-components for simplicity/avoiding encoder list output
class ResEncoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.down_t = down_t
        self.stride_t = stride_t
        block_kwargs_copy = dict(**block_kwargs)
        self.encoder_block = EncoderConvBlock(
            input_emb_width,
            output_emb_width,
            down_t,
            stride_t,
            **block_kwargs_copy,
        )

    def forward(self, x):
        x = self.encoder_block(x)
        return x


class ResDecoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.down_t = down_t
        self.stride_t = stride_t
        self.decoder_block = DecoderConvBlock(
            output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs
        )
        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, x):
        x = self.decoder_block(x)
        x = self.out(x)
        return x


# Returns latent dimension, resnet block configurations and the whole ResAeConfig object
def _res_ae_configs(
    hyper_cfg: HyperConfig,
) -> tuple[int, dict[int], ResAeConfig]:
    res_ae_config = hyper_cfg.res_ae
    assert (
        len(res_ae_config.downs_t)
        == len(res_ae_config.strides_t)
        == res_ae_config.levels
    ), "Mismatch in res_ae levels configurations"
    block_kwargs = dict(
        width=res_ae_config.block_width,
        depth=res_ae_config.block_depth,
        m_conv=res_ae_config.block_m_conv,
        dilation_cycle=res_ae_config.block_dilation_cycle,
        dilation_growth_rate=res_ae_config.block_dilation_growth_rate,
    )
    return hyper_cfg.latent_dim, block_kwargs, res_ae_config


# For single-level res-encoders
def res_encoder_output_seq_length(hyper_cfg: HyperConfig) -> int:
    res_ae_cfg = hyper_cfg.res_ae
    assert (
        res_ae_cfg.levels == 1
    ), f"Method only supported for single-level but number of levels was {res_ae_cfg.levels}"
    return hyper_cfg.seq_len // (res_ae_cfg.strides_t[0] ** res_ae_cfg.downs_t[0])


def get_res_encoder(hyper_cfg: HyperConfig) -> ResEncoder:
    latent_dim, block_kwargs, res_ae_cfg = _res_ae_configs(hyper_cfg)
    assert (
        res_ae_cfg.levels == 1
    ), f"Method only supported for single-level but number of levels was {res_ae_cfg.levels}"
    return ResEncoder(
        res_ae_cfg.input_emb_width,
        latent_dim,
        res_ae_cfg.downs_t[0],
        res_ae_cfg.strides_t[0],
        **block_kwargs,
    )


def get_res_decoder(hyper_cfg: HyperConfig) -> ResDecoder:
    latent_dim, block_kwargs, res_ae_cfg = _res_ae_configs(hyper_cfg)
    assert (
        res_ae_cfg.levels == 1
    ), f"Method only supported for single-level but number of levels was {res_ae_cfg.levels}"
    return ResDecoder(
        res_ae_cfg.input_emb_width,
        latent_dim,
        res_ae_cfg.downs_t[0],
        res_ae_cfg.strides_t[0],
        **block_kwargs,
    )


def _create_res_autoencoder(hyper_cfg: HyperConfig) -> AutoEncoder:
    encoder = get_res_encoder(hyper_cfg)
    decoder = get_res_decoder(hyper_cfg)
    return AutoEncoder(encoder, decoder)


def _create_autoencoder(hyper_cfg: HyperConfig) -> AutoEncoder:
    encoder = Encoder(hyper_cfg.seq_len, hyper_cfg.latent_dim)
    decoder = Decoder(hyper_cfg.seq_len, hyper_cfg.latent_dim, encoder.output_lengths)
    return AutoEncoder(encoder, decoder)


def get_autoencoder(hyper_cfg: HyperConfig) -> AutoEncoder:
    if hyper_cfg.model == "ae":
        return _create_autoencoder(hyper_cfg)
    elif hyper_cfg.model == "res-ae":
        return _create_res_autoencoder(hyper_cfg)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(hyper_cfg.model))

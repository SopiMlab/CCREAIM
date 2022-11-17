from torch import nn

from ..utils import util
from .resnet import Resnet1D

"""
Semi-hardcoded AutoEncoder implementation

Possible TODOs:
- Make more hardcoded models
AND/OR
- configurable model structures
"""

# Returns an nn.Conv1d layer according to given parameters, with proper padding calculated for
# output_length to be ceil(input_length/stride)
def create_conv1d_layer(
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    input_length: int,
):
    padding, length_out = util.conf_same_padding_calc(input_length, stride, kernel_size)
    return (
        nn.Conv1d(
            input_channels, output_channels, kernel_size, stride=stride, padding=padding
        ),
        length_out,
    )


class Encoder(nn.Module):
    def __init__(self, seq_length: int, latent_dim: int):
        super().__init__()
        # The negative slope coefficient for leaky ReLU
        leaky_relu_alpha = 0.2

        # Record the output lengths of the layers for decoder
        self.output_lengths = []

        # First layer
        self.conv1, len_out = create_conv1d_layer(
            input_channels=1,
            output_channels=64,
            kernel_size=7,
            stride=1,
            input_length=seq_length,
        )
        self.b_norm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(leaky_relu_alpha)
        self.output_lengths.append(len_out)

        # Second layer
        self.conv2, len_out = create_conv1d_layer(
            input_channels=64,
            output_channels=128,
            kernel_size=5,
            stride=2,
            input_length=len_out,
        )
        self.b_norm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.LeakyReLU(leaky_relu_alpha)
        self.output_lengths.append(len_out)

        # Third layer
        self.conv3, len_out = create_conv1d_layer(
            input_channels=128,
            output_channels=256,
            kernel_size=9,
            stride=4,
            input_length=len_out,
        )
        self.b_norm3 = nn.BatchNorm1d(256)
        self.relu3 = nn.LeakyReLU(leaky_relu_alpha)
        self.output_lengths.append(len_out)

        # Fourth layer
        self.conv4, len_out = create_conv1d_layer(
            input_channels=256,
            output_channels=512,
            kernel_size=9,
            stride=4,
            input_length=len_out,
        )
        self.b_norm4 = nn.BatchNorm1d(512)
        self.relu4 = nn.LeakyReLU(leaky_relu_alpha)
        self.output_lengths.append(len_out)

        # Fifth layer
        self.conv5, len_out = create_conv1d_layer(
            input_channels=512,
            output_channels=1024,
            kernel_size=9,
            stride=4,
            input_length=len_out,
        )
        self.relu5 = nn.LeakyReLU(leaky_relu_alpha)
        self.output_lengths.append(len_out)

        # Final layer
        self.conv6, len_out = create_conv1d_layer(
            input_channels=1024,
            output_channels=latent_dim,
            kernel_size=5,
            stride=1,
            input_length=len_out,
        )
        self.output_lengths.append(len_out)

    def forward(self, data):

        data = self.conv1(data)
        data = self.b_norm1(data)
        data = self.relu1(data)

        data = self.conv2(data)
        data = self.b_norm2(data)
        data = self.relu2(data)

        data = self.conv3(data)
        data = self.b_norm3(data)
        data = self.relu3(data)

        data = self.conv4(data)
        data = self.b_norm4(data)
        data = self.relu4(data)

        data = self.conv5(data)
        data = self.relu5(data)

        data = self.conv6(data)

        return data


# Creates an nn.ConvTranspose1d layer according to given parameters, where a the correct
# padding and output_padding is used for a given input_length => output_length mapping
def create_convtranspose1d_layer(
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    input_length: int,
    output_length: int,
):
    padding, output_padding = util.conf_same_padding_calc_t(
        input_length, output_length, stride, kernel_size
    )
    return nn.ConvTranspose1d(
        input_channels,
        output_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )


class Decoder(nn.Module):
    def __init__(self, seq_length: int, latent_dim: int, output_lengths: list[int]):
        super().__init__()

        # The negative slope coefficient for leaky ReLU
        leaky_relu_alpha = 0.2

        # Output lengths for getting correct paddings
        # that reflect the encoder's sequence lengths
        len_outputs = output_lengths.copy()

        # First layer
        len_in = len_outputs.pop()
        len_out = len_outputs.pop()
        self.conv6 = create_convtranspose1d_layer(
            input_channels=latent_dim,
            output_channels=1024,
            kernel_size=5,
            stride=1,
            input_length=len_in,
            output_length=len_out,
        )
        self.relu5 = nn.LeakyReLU(leaky_relu_alpha)

        # Second layer
        len_in = len_out
        len_out = len_outputs.pop()
        self.conv5 = create_convtranspose1d_layer(
            input_channels=1024,
            output_channels=512,
            kernel_size=9,
            stride=4,
            input_length=len_in,
            output_length=len_out,
        )
        self.relu4 = nn.LeakyReLU(leaky_relu_alpha)

        # Third layer
        len_in = len_out
        len_out = len_outputs.pop()
        self.conv4 = create_convtranspose1d_layer(
            input_channels=512,
            output_channels=256,
            kernel_size=9,
            stride=4,
            input_length=len_in,
            output_length=len_out,
        )
        self.relu3 = nn.LeakyReLU(leaky_relu_alpha)

        # Fourth layer
        len_in = len_out
        len_out = len_outputs.pop()
        self.conv3 = create_convtranspose1d_layer(
            input_channels=256,
            output_channels=128,
            kernel_size=9,
            stride=4,
            input_length=len_in,
            output_length=len_out,
        )
        self.relu2 = nn.LeakyReLU(leaky_relu_alpha)

        # Fifth layer
        len_in = len_out
        len_out = len_outputs.pop()
        self.relu1 = nn.LeakyReLU(leaky_relu_alpha)
        self.conv2 = create_convtranspose1d_layer(
            input_channels=128,
            output_channels=64,
            kernel_size=5,
            stride=2,
            input_length=len_in,
            output_length=len_out,
        )

        # Final layer
        len_in = len_out
        len_out = seq_length
        self.conv1 = create_convtranspose1d_layer(
            input_channels=64,
            output_channels=1,
            kernel_size=7,
            stride=1,
            input_length=len_in,
            output_length=len_out,
        )

    def forward(self, data):

        data = self.conv6(data)
        data = self.relu5(data)

        data = self.conv5(data)
        data = self.relu4(data)

        data = self.conv4(data)
        data = self.relu3(data)

        data = self.conv3(data)
        data = self.relu2(data)

        data = self.conv2(data)
        data = self.relu1(data)

        data = self.conv1(data)

        return data


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.net = nn.Sequential(self.encoder, self.decoder)

    def forward(self, input_data):
        return self.net(input_data)

    def encode(self, input_data):
        return self.encoder(input_data)


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


# For both Encoder and Decoder:
# input_emb_width = seq_len
# output_emb_width = latent_dim
class ResEncoder(nn.Module):
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


class ResDecoder(nn.Module):
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


def _create_res_autoencoder(latent_dim):
    levels = 1  # orig 3
    downs_t = [3]  # orig (3, 2, 2)
    strides_t = [2]  # orig (2, 2, 2)
    input_emb_width = 1
    output_emb_width = latent_dim  # orig 64
    block_width = 32
    block_depth = 2
    block_m_conv = 1.0
    block_dilation_growth_rate = 3
    block_dilation_cycle = None

    block_kwargs = dict(
        width=block_width,
        depth=block_depth,
        m_conv=block_m_conv,
        dilation_cycle=block_dilation_cycle,
    )

    encoder = ResEncoder(
        input_emb_width, output_emb_width, levels, downs_t, strides_t, **block_kwargs
    )
    decoder = ResDecoder(
        input_emb_width, output_emb_width, levels, downs_t, strides_t, **block_kwargs
    )
    return AutoEncoder(encoder, decoder)


def _create_autoencoder(seq_length: int, latent_dim: int):
    encoder = Encoder(seq_length, latent_dim)
    decoder = Decoder(seq_length, latent_dim, encoder.output_lengths)
    return AutoEncoder(encoder, decoder)


def get_autoencoder(name: str, seq_length: int, latent_dim: int):
    if name == "base":
        return _create_autoencoder(seq_length, latent_dim)
    elif name == "res-ae":
        return _create_res_autoencoder(latent_dim)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

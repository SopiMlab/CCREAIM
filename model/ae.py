from torch import nn

from model.resnet import Resnet1D
from utils import util

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
                        zero_out,
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
    def __init__(self):
        super().__init__()


class ResEncoder(nn.Module):
    def __init__(self):
        super().__init__()


class ResDecoder(nn.Module):
    def __init__(self):
        super().__init__()


class ResAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()


def _create_autoencoder(seq_length: int, latent_dim: int):
    encoder = Encoder(seq_length, latent_dim)
    decoder = Decoder(seq_length, latent_dim, encoder.output_lengths)
    return AutoEncoder(encoder, decoder)


def get_autoencoder(name: str, seq_length: int, latent_dim: int):
    if name == "base":
        return _create_autoencoder(seq_length, latent_dim)
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

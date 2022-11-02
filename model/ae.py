from torch import nn

"""
Hard-coded Decoder and Encoder implementations for specific sized input values

The signal sequence length of the output of convolutional layers follow these
equations:
    Conv1d:
    L_out = floor((L_in​ + 2*padding − dilation*(kernel_size − 1) − 1)/stride ​+ 1)

    ConvTranspose1d:
    L_out = (L_in ​− 1)*stride − 2*padding + dilation*(kernel_size − 1) + output_padding + 1

The paddings were decided by trying to keep L_out the same as it was on the original tf implementation.
For the Decoder, the Encoder's paddings were mirrored and then tuned with the 'output_padding' argument.

Possible TODOs:
- Make more hardcoded models
AND/OR
- Parametrize as much as possible:
    - calculate the required paddings automatically
    - configurable model structures
"""


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # The negative slope coefficient for leaky ReLU
        leaky_relu_alpha = 0.2
        latent_dim = 128
        # [input shape = (1, 1, 1000)]
        # First layer [output shape = (1, 64, 1000)]
        self.conv1 = nn.Conv1d(1, 64, 7, padding=3)
        self.b_norm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(leaky_relu_alpha)

        # Second layer [output shape = (1, 128, 500)]
        self.conv2 = nn.Conv1d(64, 128, 5, stride=2, padding=2)
        self.b_norm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.LeakyReLU(leaky_relu_alpha)

        # Third layer [output shape = (1, 256, 125)]
        self.conv3 = nn.Conv1d(128, 256, 9, stride=4, padding=3)
        self.b_norm3 = nn.BatchNorm1d(256)
        self.relu3 = nn.LeakyReLU(leaky_relu_alpha)

        # Fourth layer [output shape = (1, 512, 32)]
        self.conv4 = nn.Conv1d(256, 512, 9, stride=4, padding=4)
        self.b_norm4 = nn.BatchNorm1d(512)
        self.relu4 = nn.LeakyReLU(leaky_relu_alpha)

        # Fifth layer [output shape = (1, 1024, 8)]
        self.conv5 = nn.Conv1d(512, 1024, 9, stride=4, padding=3)
        self.relu5 = nn.LeakyReLU(leaky_relu_alpha)

        # Final layer [output shape = (1, latent_dim*2, 8)]
        self.conv6 = nn.Conv1d(1024, latent_dim * 2, 5, padding=2)

    def forward(self, data):
        # print("input data shape: {}".format(data.size()))

        data = self.conv1(data)
        data = self.b_norm1(data)
        data = self.relu1(data)
        # print("data shape after layer 1: {}".format(data.size()))

        data = self.conv2(data)
        data = self.b_norm2(data)
        data = self.relu2(data)
        # print("data shape after layer 2: {}".format(data.size()))

        data = self.conv3(data)
        data = self.b_norm3(data)
        data = self.relu3(data)
        # print("data shape after layer 3: {}".format(data.size()))

        data = self.conv4(data)
        data = self.b_norm4(data)
        data = self.relu4(data)
        # print("data shape after layer 4: {}".format(data.size()))

        data = self.conv5(data)
        data = self.relu5(data)
        # print("data shape after layer 5: {}".format(data.size()))

        data = self.conv6(data)
        # print("data shape after layer 6: {}".format(data.size()))

        return data


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # The negative slope coefficient for leaky ReLU
        leaky_relu_alpha = 0.2
        latent_dim = 128

        # [latent shape = (1, latent_dim*2, 8)]
        # First layer [output shape = (1, 1024, 8)]
        self.conv6 = nn.ConvTranspose1d(latent_dim * 2, 1024, 5, padding=2)
        self.relu5 = nn.LeakyReLU(leaky_relu_alpha)

        # Second layer [output shape = (1, 512, 32)]
        self.conv5 = nn.ConvTranspose1d(
            1024, 512, 9, stride=4, padding=3, output_padding=1
        )
        self.relu4 = nn.LeakyReLU(leaky_relu_alpha)

        # Third layer [output shape = (1, 256, 125)]
        self.conv4 = nn.ConvTranspose1d(512, 256, 9, stride=4, padding=4)
        self.relu3 = nn.LeakyReLU(leaky_relu_alpha)

        # Fourth layer [output shape = (1, 128, 500)]
        self.conv3 = nn.ConvTranspose1d(
            256, 128, 9, stride=4, padding=3, output_padding=1
        )
        self.relu2 = nn.LeakyReLU(leaky_relu_alpha)

        # Fifth layer [output shape = (1, 64, 1000)]
        self.conv2 = nn.ConvTranspose1d(
            128, 64, 5, stride=2, padding=2, output_padding=1
        )
        self.relu1 = nn.LeakyReLU(leaky_relu_alpha)

        # Final layer [output shape = (1, 1, 1000)]
        self.conv1 = nn.ConvTranspose1d(64, 1, 7, padding=3)

    def forward(self, data):
        # print("latent data shape: {}".format(data.size()))

        data = self.conv6(data)
        data = self.relu5(data)
        # print("data shape after layer 1: {}".format(data.size()))

        data = self.conv5(data)
        data = self.relu4(data)
        # print("data shape after layer 2: {}".format(data.size()))

        data = self.conv4(data)
        data = self.relu3(data)
        # print("data shape after layer 3: {}".format(data.size()))

        data = self.conv3(data)
        data = self.relu2(data)
        # print("data shape after layer 4: {}".format(data.size()))

        data = self.conv2(data)
        data = self.relu1(data)
        # print("data shape after layer 5: {}".format(data.size()))

        data = self.conv1(data)
        # print("data shape after layer 6: {}".format(data.size()))

        return data


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.net = nn.Sequential(encoder, decoder)
        # TODO parameterise these
        self.loss_fn = nn.MSELoss()

    def forward(self, input_data):
        return self.net(input_data)


def get_autoencoder(name: str):
    if name == "base":
        return AutoEncoder(Encoder(), Decoder())
    else:
        raise ValueError("Unknown autoencoder name: '{}'".format(name))

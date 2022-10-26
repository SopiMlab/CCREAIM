from dataclasses import dataclass

from torch import nn

"""
NOTICE!
padding="same" for all the convolutional layers in the tf implementation, but
this was not supported for nn.ConvTranspose1d strided convolutions, so replaced with
padding="zero" at least for now
"""


@dataclass
class EncoderConfig:
    """Class to save encoder configuration values"""

    INPUT_SIZE: int
    LATENT_SIZE: int

    """
    Contains the structure for the hidden layers, each represented by a dict
    with keys "input_ch", "output_ch", "kernel_size" and "stride"
    """
    HIDDEN_LAYER_SPECS: list[dict[str, int]]

    """
    The negative slope coefficient for leaky ReLU
    """
    LEAKY_RELU_ALPHA: float = 0.2


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        modules = []

        # Create the first layer separately to allow changing config.INPUT_SIZE freely
        first_hidden_layer_input_ch = config.HIDDEN_LAYER_SPECS[0]["input_ch"]
        input_layer = nn.Conv1d(
            config.INPUT_SIZE, first_hidden_layer_input_ch, 7
        )  # kernel_size = from the old tf implementation
        modules.append(input_layer)
        modules.append(nn.BatchNorm1d(first_hidden_layer_input_ch))
        modules.append(nn.LeakyReLU(config.LEAKY_RELU_ALPHA))

        # Hidden layers
        # Loop through all the hidden layers except the last
        for layer_specs in config.HIDDEN_LAYER_SPECS[:-1]:

            input_ch = layer_specs["input_ch"]
            output_ch = layer_specs["output_ch"]
            kernel_size = layer_specs["kernel_size"]
            stride = layer_specs["stride"]

            modules.append(nn.Conv1d(input_ch, output_ch, kernel_size, stride=stride))
            modules.append(nn.BatchNorm1d(output_ch))
            modules.append(nn.LeakyReLU(config.LEAKY_RELU_ALPHA))

        # Last hidden layer doesn't include batch normalization, so it's created outside the loop
        last_input_ch = config.HIDDEN_LAYER_SPECS[-1]["input_ch"]
        last_output_ch = config.HIDDEN_LAYER_SPECS[-1]["output_ch"]
        last_kernel_size = config.HIDDEN_LAYER_SPECS[-1]["kernel_size"]
        last_stride = config.HIDDEN_LAYER_SPECS[-1]["stride"]

        modules.append(
            nn.Conv1d(last_input_ch, last_output_ch, last_kernel_size, last_stride)
        )
        modules.append(nn.LeakyReLU(config.LEAKY_RELU_ALPHA))

        # Final layer separately
        modules.append(
            nn.Conv1d(last_output_ch, config.LATENT_SIZE, 5)
        )  # kernel_size = 5 from old tf implementation

        self.net = nn.Sequential(*modules)

    def forward(self, input_data):
        return self.net(input_data)


@dataclass
class DecoderConfig:
    """Class to save decoder configuration values"""

    LATENT_SIZE: int

    """
    Contains the structure for the hidden layers, each represented by a dict
    with keys "input_ch", "output_ch", "kernel_size" and "stride"
    """
    HIDDEN_LAYER_SPECS: list[dict[str, int]]

    LEAKY_RELU_ALPHA: float = 0.2


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        modules = []

        # Create the first layer separately to allow changing config.LATENT_SIZE freely
        first_hidden_layer_input_ch = config.HIDDEN_LAYER_SPECS[0]["input_ch"]
        input_layer = nn.Conv1d(
            config.LATENT_SIZE, first_hidden_layer_input_ch, 5
        )  # kernel_size = 5 from old tf implementation
        modules.append(input_layer)
        modules.append(nn.LeakyReLU(config.LEAKY_RELU_ALPHA))

        # Hidden layers
        # Loop through all the hidden layers except the last
        for layer_specs in config.HIDDEN_LAYER_SPECS[:-1]:

            input_ch = layer_specs["input_ch"]
            output_ch = layer_specs["output_ch"]
            kernel_size = layer_specs["kernel_size"]
            stride = layer_specs["stride"]

            modules.append(
                nn.ConvTranspose1d(input_ch, output_ch, kernel_size, stride=stride)
            )
            modules.append(nn.LeakyReLU(config.LEAKY_RELU_ALPHA))

        # Final layer
        last_output_ch = config.HIDDEN_LAYER_SPECS[-1]["output_ch"]
        modules.append(
            nn.ConvTranspose1d(last_output_ch, 1, 7)
        )  # kernel_size = from the old tf implementation

        self.net = nn.Sequential(*modules)

    def forward(self, latent_input_data):
        return self.net(latent_input_data)


class AutoEncoder(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.net = nn.Sequential(Encoder(encoder_config), Decoder(decoder_config))

    def forward(self, input_data):
        return self.net(input_data)

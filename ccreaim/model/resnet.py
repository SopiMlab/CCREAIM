import math

from torch import nn


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                n_in,
                n_state,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        m_conv=1.0,
        dilation_growth_rate=1,
        dilation_cycle=None,
        res_scale=False,
    ):
        super().__init__()

        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle

        blocks = [
            ResConv1DBlock(
                n_in,
                int(m_conv * n_in),
                dilation=dilation_growth_rate ** _get_depth(depth),
                res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth),
            )
            for depth in range(n_depth)
        ]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

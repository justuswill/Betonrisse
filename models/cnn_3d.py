import torch
import torch.nn as nn
import torch.nn.functional as F


"""
3D CNN for Image Classification
"""


class Net(nn.Module):
    def __init__(self, in_channels=1, dim=100, layers=1, out_conv_channels=None, extra_pool=None, dropout=0):
        """
        Define a standard CNN with <layer> blocks of convolution + pooling
        and a final pooling layer with outdim ~ <dim> / 2**(<layers> + <extrapool>)
        followed by three linear layers

        :param in_channels: (color) channels of the image, default: 1 (grayscale)
        :param dim: shape of the image as (dim x dim x dim), default: 100
        :param layers: number of convolutional layers, default: 1
        :param out_conv_channels: number of conv. channels in the last layer, default: 16 * 2**layers
        :param extra_pool: size of additional pooling after last conv+pool block, default:  2**(4 - layers)
        :param dropout: dropout percentage in all but last layer
        """
        super().__init__()
        # use channel sizes [32, 64, 128, ...] and outdim ~ dim/16
        if out_conv_channels is None:
            out_conv_channels = 2 ** (4 + layers)
        if extra_pool is None:
            extra_pool = 2 ** max(0, 4 - layers)

        self.layers = layers
        conv_channels = [in_channels] + [int(out_conv_channels / 2**k) for k in range(self.layers - 1, -1, -1)]

        # Convolutions with Batch Normalization - outdim = dim + (3 - kern)
        for i in range(self.layers):
            conv = nn.Sequential(
                nn.Conv3d(
                    in_channels=conv_channels[i], out_channels=conv_channels[i+1],
                    kernel_size=4, stride=1, padding=1, bias=False
                ),
                nn.BatchNorm3d(conv_channels[i+1]),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.add_module("conv%d" % i, conv)

        # Pooling - outdim = dim / 2
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.final_pool = nn.MaxPool3d(kernel_size=2 * extra_pool, stride=2**(extra_pool + 1))

        # Compute output size after conv+pool layers
        self.out_dim = dim
        for _ in range(layers):
            self.out_dim = (self.out_dim - 1) // 2
        self.out_dim = self.out_dim // extra_pool

        # Linear with Dropout in all but last layer
        self.fc1 = nn.Sequential(nn.Linear(out_conv_channels * self.out_dim ** 3, 128), nn.Dropout(p=dropout))
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.Dropout(p=dropout))
        self.fc3 = nn.Sequential(nn.Linear(64, 1), nn.Dropout(p=0))

    def forward(self, x):
        for i in range(self.layers):
            conv = getattr(self, "conv%d" % i)
            pool = self.final_pool if i == self.layers - 1 else self.pool
            x = pool(conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




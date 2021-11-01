import torch
import torch.nn as nn
import torch.nn.functional as F


# Old net models

class Net(nn.Module):
    def __init__(self, in_channels=1, dim=100, out_conv_channels=512):
        super().__init__()

        conv1_channels = int(out_conv_channels / 8)
        conv2_channels = int(out_conv_channels / 4)
        conv3_channels = int(out_conv_channels / 2)

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels, out_channels=conv1_channels, kernel_size=4, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=4,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=4,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv3_channels, out_channels=out_conv_channels, kernel_size=4,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        f = lambda x: (x - 1) // 2
        self.out_dim = f(f(f(f(dim))))

        self.fc1 = nn.Linear(out_conv_channels * self.out_dim ** 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    def __init__(self, in_channels=1, dim=100, layers=4, out_conv_channels=None, extra_pool=None):
        """
        Define a standard CNN with <layer> blocks of convolution + pooling
        and a final pooling layer with outdim ~ dim/2**(layers+extrapool)
        """
        super().__init__()
        # use channel sizes [1, 32, 64, 128, ...] and outdim ~ dim/16
        if out_conv_channels is None:
            out_conv_channels = 2 ** (4 + layers)
        if extra_pool is None:
            extra_pool = max(0, 4 - layers)

        # Convolutions - outdim = dim + (3 - kern)
        self.layers = layers
        conv_channels = [in_channels] + [int(out_conv_channels / 2**k) for k in range(self.layers - 1, -1, -1)]
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
        self.final_pool = nn.MaxPool3d(kernel_size=2**(extra_pool + 1), stride=2**(extra_pool + 1))

        self.out_dim = dim
        for _ in range(layers):
            self.out_dim = (self.out_dim - 1) // 2
        self.out_dim = self.out_dim // 2**extra_pool

        self.fc1 = nn.Linear(out_conv_channels * self.out_dim ** 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

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


class Net(nn.Module):
    def __init__(self, in_channels=1, dim=100, layers=4, out_conv_channels=None, extra_pool=None, dropout=0):
        """
        Define a standard CNN with <layer> blocks of convolution + pooling
        and a final pooling layer with outdim ~ dim/2**(layers+extrapool)
        """
        super().__init__()
        # use channel sizes [1, 32, 64, 128, ...] and outdim ~ dim/16
        if out_conv_channels is None:
            out_conv_channels = 2 ** (4 + layers)
        if extra_pool is None:
            extra_pool = max(0, 4 - layers)

        # Convolutions - outdim = dim + (3 - kern)
        self.layers = layers
        conv_channels = [in_channels] + [int(out_conv_channels / 2**k) for k in range(self.layers - 1, -1, -1)]
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
        self.final_pool = nn.MaxPool3d(kernel_size=2**(extra_pool + 1), stride=2**(extra_pool + 1))

        self.out_dim = dim
        for _ in range(layers):
            self.out_dim = (self.out_dim - 1) // 2
        self.out_dim = self.out_dim // 2**extra_pool

        self.fc1 = nn.Sequential(nn.Linear(out_conv_channels * self.out_dim ** 3, 128), nn.Dropout(p=dropout))
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.Dropout(p=dropout))
        self.fc3 = nn.Sequential(nn.Linear(64, 1), nn.Dropout(p=dropout))

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
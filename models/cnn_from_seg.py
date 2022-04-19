import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def conv_block3d(in_channels, out_channels, activation, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm3d(out_channels),
        activation
    )


def max_pooling3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)):
    return nn.MaxPool3d(kernel_size=kernel_size, stride=stride)


# activation = nn.ReLU()
# model = Unet3d(1, 1, model_depth=3, num_filters = 16, activation=activation, final_activation=nn.Sigmoid())
class Unet3d(nn.Module):
    def __init__(self, in_channels=1, dim=100, layers=2, num_filters=16, activation=nn.ReLU(), dropout=0, freeze=True):
        super(Unet3d, self).__init__()
        self.in_channels = in_channels
        self.layers = layers
        self.num_filters = num_filters
        self.activation = activation
        self.dropout = dropout

        self.first_conv = conv_block3d(self.in_channels, self.num_filters, self.activation)
        if freeze:
            for x in self.first_conv.parameters():
                x.requires_grad = False
        if layers > 1:
            self.first_conv2 = conv_block3d(self.num_filters, self.num_filters, self.activation)
            if freeze:
                for x in self.first_conv2.parameters():
                    x.requires_grad = False
        if layers > 2:
            self.pool = max_pooling3d()
            self.conv0 = conv_block3d(self.num_filters, self.num_filters * 2, self.activation)
            dim = dim // 2
            self.num_filters = 2 * self.num_filters
            if freeze:
                for x in self.conv0.parameters():
                    x.requires_grad = False

        # k = dim - 2 ** int(np.log2(dim))
        # s = 2 ** (int(np.log2(dim)) - 3)
        # assert k >= s >= 1
        k = 16 if layers <= 2 else 8
        pad = int(np.ceil(dim % k) / 2)
        self.final_pool = nn.MaxPool3d(kernel_size=k, stride=k, padding=pad)
        self.final_dim = (dim + 2 * pad - k) // k + 1

        # Linear with Dropout in all but last layer
        self.fc1 = nn.Sequential(nn.Linear(self.num_filters * self.final_dim ** 3, 128), nn.Dropout(p=dropout))
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.Dropout(p=dropout))
        self.fc3 = nn.Sequential(nn.Linear(64, 1), nn.Dropout(p=0))

    def forward(self, x, filter=False):
        f = self.first_conv(x)
        if self.layers > 1:
            f = self.first_conv2(f)
        if self.layers > 2:
            f = self.pool(f)
            f = self.conv0(f)
        x = self.final_pool(f)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if filter:
            return f, x
        else:
            return x

    def load_state_dict(self, state_dict):
        """ Partially load Unet parameters """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("first_conv.0") or k.startswith("first_conv.1"):
                new_state_dict[k] = v
            elif (k.startswith("first_conv.3") or k.startswith("first_conv.4")) and self.layers > 1:
                new_state_dict["first_conv2.%d" % (int(k.split(".")[1]) - 2)] = v
            elif (k.startswith("encoders.0.conv_block.0") or k.startswith("encoders.0.conv_block.1")) and self.layers > 2:
                new_state_dict[k.replace("encoders.0.conv_block", "conv0")] = v
        super().load_state_dict(new_state_dict, strict=False)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch
import torch.nn as nn
from torchsummary import summary

import random
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from data.data import Betondata
from data.data_tools import ToTensor, normalize, random_rotate_flip_3d


"""
Network architecture based on paper
https://arxiv.org/abs/1606.06650
with code from
https://github.com/JielongZ/3D-UNet-PyTorch-Implementation
"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        # x = self.conv3d(x)
        x = F.elu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 16
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                if depth == 0:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
                print(k, x.shape)
                if k.endswith("1"):
                    down_sampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)
                print(k, x.shape)

        return x, down_sampling_features


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 16
        # user nn.ModuleDict() to store ops
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            # print(depth)
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            # print(feat_map_channels * 4)
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, down_sampling_features):
        """
        :param x: inputs
        :param down_sampling_features: feature maps from encoder path
        :return: output
        """
        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                x = op(x)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="sigmoid"):
        super(Unet, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.sigmoid(x)
        print("Final output shape: ", x.shape)
        return x


def test_unet3d():
    dim = 64  # cube volume
    net = Unet(in_channels=1, out_channels=512, model_depth=4)
    noise = torch.rand(1, 1, dim, dim, dim)

    out = net(noise)
    print("Output shape", out.shape)


def train_unet3d(img_dirs=None, load="", checkpoints=True, num_epochs=5):
    """
    Train a 3D-Gan to generate more data

    :param img_dirs: path (or list of paths) to image npy files
    :param load: if not "", load Unet from here and continue training
    :param checkpoints: if state of Unet should be saved after each epoch
    :param num_epochs: number of epochs to train
    """

    # Seed
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if img_dirs is not None:
        data = Betondata(img_dirs=img_dirs, transform=transforms.Compose([
            transforms.Lambda(ToTensor()),
            transforms.Lambda(normalize(33.24, 6.69)),
            transforms.Lambda(random_rotate_flip_3d())
        ]))
    else:
        data = Synthdata()
    dataloader = DataLoader(data, batch_size=4, shuffle=True, num_workers=1)

    # CNNs
    noise_dim = 200
    in_channels = 512
    dim = 100
    netG = Generator(in_channels=512, out_dim=dim, out_channels=1, noise_dim=noise_dim).to(device)
    netD = Discriminator(in_channels=1, dim=dim, out_conv_channels=in_channels).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Load state if possible
    if loadG != '':
        try:
            netG.load_state_dict(torch.load(loadG))
        except FileNotFoundError:
            print("No Generator loaded")
            pass
    if loadD != '':
        try:
            netD.load_state_dict(torch.load(loadD))
        except FileNotFoundError:
            print("No Discriminator loaded")
            pass

    net = Unet(in_channels=1, out_channels=512)
    #optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    #criterion = DiceLoss()

    # self.net.train()
    pet_paths = data_paths_loader(self.data_dir, self.modalities[0])
    print(pet_paths)
    mask_paths = data_paths_loader(self.data_dir, self.modalities[1])
    pets, masks = dataset_loader(pet_paths, mask_paths)
    training_steps = len(pets) // self.batch_size

    for epoch in range(self.no_epochs):
        start_time = time.time()
        train_losses, train_iou = 0, 0
        for step in range(training_steps):
            print("Training step {}".format(step))

            x_batch, y_batch = batch_data_loader(pets, masks, iter_step=step, batch_size=self.batch_size)
            x_batch = torch.from_numpy(x_batch).cuda()
            y_batch = torch.from_numpy(y_batch).cuda()

            self.optimizer.zero_grad()

            logits = self.net(x_batch)
            y_batch = y_batch.type(torch.int8)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()
            # train_iou += mean_iou(y_batch, logits)
            train_losses += loss.item()
        end_time = time.time()
        print("Epoch {}, training loss {:.4f}, time {:.2f}".format(epoch, train_losses / training_steps,
                                                                   end_time - start_time))


if __name__ == "__main__":
    test_unet3d()
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from data import Betondata, plot_batch, ToTensor, Normalize, Random_rotate_flip_3d
from paths import *

plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH


"""
GAN + train, for generation of additional training data (needs more work)

Currently shows mode collapse, potential fixes with:
batch normalization between G and D, random rotate/flip, label smoothing, cut vs pad

Network architecture based on original paper NeurIPS 2016
https://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf
with code from
https://github.com/black0017/3D-GAN-pytorch
Training etc based on
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
Debugging using (in part)
https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/
https://stackoverflow.com/questions/44313306/dcgans-discriminator-getting-too-strong-too-quickly-to-allow-generator-to-learn
"""


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=3, dim=64, out_conv_channels=512):
        super(Discriminator, self).__init__()
        conv1_channels = int(out_conv_channels / 8)
        conv2_channels = int(out_conv_channels / 4)
        conv3_channels = int(out_conv_channels / 2)
        self.out_conv_channels = out_conv_channels
        self.out_dim = int(dim // 16)

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels, out_channels=conv1_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv3_channels, out_channels=out_conv_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(out_conv_channels * self.out_dim * self.out_dim * self.out_dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply linear + sigmoid
        x = x.view(-1, self.out_conv_channels * self.out_dim * self.out_dim * self.out_dim)
        x = self.out(x)
        return x


class Generator_padding(torch.nn.Module):
    def __init__(self, in_channels=512, out_dim=64, out_channels=1, noise_dim=200, activation="sigmoid"):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Allow outputs that are not dividable by 16 by cutting
        # alternative is output padding
        self.in_dim = math.ceil(out_dim / 16)
        dif_16 = 16 * self.in_dim - self.out_dim
        self.cut1 = dif_16 % 16 // 8
        self.cut2 = dif_16 % 8 // 4
        self.cut3 = dif_16 % 4 // 2
        self.cut4 = dif_16 % 2 // 1

        conv1_out_channels = int(self.in_channels / 2.0)
        conv2_out_channels = int(conv1_out_channels / 2)
        conv3_out_channels = int(conv2_out_channels / 2)

        self.linear = torch.nn.Linear(noise_dim, self.in_channels * self.in_dim * self.in_dim * self.in_dim)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm3d(conv1_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm3d(conv2_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv2_out_channels, out_channels=conv3_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm3d(conv3_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv3_out_channels, out_channels=out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False,
            )
        )
        if activation == "sigmoid":
            self.out = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Tanh()

    def project(self, x):
        """
        projects and reshapes latent vector to starting volume
        :param x: latent vector
        :return: starting volume
        """
        return x.view(-1, self.in_channels, self.in_dim, self.in_dim, self.in_dim)

    def cut(self, x, cut):
        """
        cuts the borders of a tensor if necessary
        """
        sz = x.size()
        return x[:, :, 0:(sz[2]-cut), 0:(sz[3]-cut), 0:(sz[4]-cut)]

    def forward(self, x):
        x = self.linear(x)
        x = self.project(x)
        x = self.conv1(x)
        x = self.cut(x, self.cut1)
        x = self.conv2(x)
        x = self.cut(x, self.cut2)
        x = self.conv3(x)
        x = self.cut(x, self.cut3)
        x = self.conv4(x)
        x = self.cut(x, self.cut4)
        return self.out(x)


class Generator(torch.nn.Module):
    def __init__(self, in_channels=512, out_dim=64, out_channels=1, noise_dim=200, activation="sigmoid"):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Allow outputs that are not dividable by 16 by cutting
        # alternative is output padding
        self.in_dim = math.ceil(out_dim / 16)
        dif_16 = 16 * self.in_dim - self.out_dim
        self.cut1 = dif_16 % 16 // 8
        self.cut2 = dif_16 % 8 // 4
        self.cut3 = dif_16 % 4 // 2
        self.cut4 = dif_16 % 2 // 1

        conv1_out_channels = int(self.in_channels / 2.0)
        conv2_out_channels = int(conv1_out_channels / 2)
        conv3_out_channels = int(conv2_out_channels / 2)

        self.linear = torch.nn.Linear(noise_dim, self.in_channels * self.in_dim * self.in_dim * self.in_dim)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=0, bias=False,
            ),
            nn.BatchNorm3d(conv1_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=0, bias=False,
            ),
            nn.BatchNorm3d(conv2_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv2_out_channels, out_channels=conv3_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=0, bias=False,
            ),
            nn.BatchNorm3d(conv3_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv3_out_channels, out_channels=out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=0, bias=False,
            )
        )
        if activation == "sigmoid":
            self.out = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Tanh()

    def project(self, x):
        """
        projects and reshapes latent vector to starting volume
        :param x: latent vector
        :return: starting volume
        """
        return x.view(-1, self.in_channels, self.in_dim, self.in_dim, self.in_dim)

    def cut(self, x, cut, pad=0):
        """
        cuts the borders of a tensor if necessary
        """
        sz = x.size()
        return x[:, :, pad:(sz[2]-cut-pad), pad:(sz[3]-cut-pad), pad:(sz[4]-cut-pad)]

    def forward(self, x):
        x = self.linear(x)
        x = self.project(x)
        x = self.conv1(x)
        x = self.cut(x, self.cut1, pad=1)
        x = self.conv2(x)
        x = self.cut(x, self.cut2, pad=1)
        x = self.conv3(x)
        x = self.cut(x, self.cut3, pad=1)
        x = self.conv4(x)
        x = self.cut(x, self.cut4, pad=1)
        return self.out(x)


def test_gan3d():
    noise_dim = 200
    in_channels = 512
    dim = 100  # cube volume
    netG = Generator(in_channels=in_channels, out_dim=dim, out_channels=1, noise_dim=noise_dim)
    noise = torch.rand(1, noise_dim)
    generated_volume = netG(noise)
    print("Generator output shape", generated_volume.shape)
    netD = Discriminator(in_channels=1, dim=dim, out_conv_channels=in_channels)
    out = netD(generated_volume)
    print("Discriminator output", out)
    summary(netG, (1, noise_dim), device="cpu")
    summary(netD, (1, 100, 100, 100), device="cpu")


def inspect_netD(path):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    in_channels = 512
    dim = 100
    netD = Discriminator(in_channels=1, dim=dim, out_conv_channels=in_channels).to(device)
    netD.load_state_dict(torch.load(path, map_location=device))
    summary(netD, (1, 100, 100, 100))
    print("Parameter means/std")
    for n, p in netD.named_parameters():
        if p.requires_grad:
            print(n, p.data.mean(), p.data.std())
    return netD


def inspect_netG(path):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    in_channels = 512
    dim = 100
    noise_dim = 400
    netG = Generator(in_channels=in_channels, out_dim=dim, out_channels=1, noise_dim=noise_dim).to(device)
    netG.load_state_dict(torch.load(path, map_location=device))

    summary(netG, (1, noise_dim))
    print("Parameter means/std")
    for n, p in netG.named_parameters():
        if p.requires_grad:
            print(n, p.data.mean(), p.data.std())

    for i in range(5):
        noise = torch.randn(8, noise_dim, device=device)
        with torch.no_grad():
            fake = netG(noise).detach().cpu()
        plot_batch(fake, 2)
        plot_batch(fake, 1)
        plot_batch(fake, 0)
        plt.show()

    return netG


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_gan3d(img_dirs, loadG="", loadD="", checkpoints=True, num_epochs=5):
    """
    Train a 3D-Gan to generate more data

    :param img_dirs: path (or list of paths) to image npy files
    :param loadG: if not "", load Generator from here and continue training
    :param loadD: same for Discriminator
    :param checkpoints: if state of Generator/Discriminator should be saved after each epoch
    :param num_epochs: number of epochs to train
    """
    # Seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    data = Betondata(img_dirs=img_dirs, transform=transforms.Compose([
        transforms.Lambda(ToTensor()),
        transforms.Lambda(Normalize(33.24, 6.69)),
        transforms.Lambda(Random_rotate_flip_3d())
    ]))
    dataloader = DataLoader(data, batch_size=4, shuffle=True, num_workers=1)

    # Nets
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

    # Loss + optimizers
    lr = 0.0002
    beta1 = 0.5
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(8, noise_dim, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            # Format batch
            real_cpu = data["X"].to(device)

            if iters % 1 == 0:
                smooth = 0.8
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                netD.zero_grad()
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), smooth * real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, noise_dim, device=device)
                # Generate fake image batch with G
                fake = netG(noise)

                # Normalize the generator output
                mean = torch.mean(fake)
                std = torch.sqrt(torch.mean((fake - mean) ** 2))
                fake = (fake - mean) / std

                label.fill_(smooth * fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()
            else:
                b_size = real_cpu.size(0)
                noise = torch.randn(b_size, noise_dim, device=device)
                fake = netG(noise)

                # Normalize the generator output
                mean = torch.mean(fake)
                std = torch.sqrt(torch.mean((fake - mean) ** 2))
                fake = (fake - mean) / std

                label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # fake labels are real for generator cost
            label.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 20 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(plot_batch(fake))
                plt.pause(0.00001)
                plt.close()

            iters += 1

        if checkpoints:
            # do checkpointing
            torch.save(netG.state_dict(), '%s_epoch_%d' % (loadG or "netG", epoch))
            torch.save(netD.state_dict(), '%s_epoch_%d' % (loadD or "netD", epoch))

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("prog.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis("off")
    Writer = FFMpegWriter(fps=5)
    Writer.setup(fig, "gan_progress.mp4", dpi=100)
    for i in img_list:
        ax.imshow(i, animated=True)
        Writer.grab_frame()
    Writer.finish()


if __name__ == "__main__":
    # test_gan3d()
    train_gan3d(img_dirs="D:Data/Beton/HPC/riss/", loadG="checkpoints/netG", loadD="checkpoints/netD", checkpoints=True, num_epochs=25)
    # inspect_netG("checkpoints/netG_epoch_0_old")

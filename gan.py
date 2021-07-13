import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from data import Betondata, plot_batch, normalize


"""
Network architecture based on original paper NeurIPS 2016
https://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf
with code from
https://github.com/black0017/3D-GAN-pytorch
Training etc based on
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
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
            nn.Linear(out_conv_channels * self.out_dim * self.out_dim * self.out_dim, 1),
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


class Generator(torch.nn.Module):
    def __init__(self, in_channels=512, out_dim=64, out_channels=1, noise_dim=200, activation="sigmoid"):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.in_dim = int(out_dim // 16)
        conv1_out_channels = int(self.in_channels / 2.0)
        conv2_out_channels = int(conv1_out_channels / 2)
        conv3_out_channels = int(conv2_out_channels / 2)

        self.linear = torch.nn.Linear(noise_dim, self.in_channels * self.in_dim * self.in_dim * self.in_dim)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False, output_padding=self.out_dim % 16 // 8,
            ),
            nn.BatchNorm3d(conv1_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False, output_padding=self.out_dim % 8 // 4,
            ),
            nn.BatchNorm3d(conv2_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv2_out_channels, out_channels=conv3_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False, output_padding=self.out_dim % 4 // 2,
            ),
            nn.BatchNorm3d(conv3_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv3_out_channels, out_channels=out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False, output_padding=self.out_dim % 2,
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

    def forward(self, x):
        x = self.linear(x)
        x = self.project(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.out(x)


def test_gan3d():
    noise_dim = 200
    in_channels = 512
    dim = 100  # cube volume
    model_generator = Generator(in_channels=512, out_dim=dim, out_channels=1, noise_dim=noise_dim)
    noise = torch.rand(1, noise_dim)
    generated_volume = model_generator(noise)
    print("Generator output shape", generated_volume.shape)
    model_discriminator = Discriminator(in_channels=1, dim=dim, out_conv_channels=in_channels)
    out = model_discriminator(generated_volume)
    print("Discriminator output", out)
    summary(model_generator, (1, noise_dim))
    summary(model_discriminator, (1, 100, 100, 100))


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
    :param: checkpoints: if state of Generator/Discriminator should be saved after each epoch
    :param num_epochs: number of epochs to train
    """
    # Seed
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = Betondata(img_dirs=img_dirs, transform=transforms.Lambda(normalize(32.69, 4.98)))
    dataloader = DataLoader(data, batch_size=4, shuffle=True, num_workers=2)

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
            pass
    if loadD != '':
        try:
            netD.load_state_dict(torch.load(loadD))
        except FileNotFoundError:
            pass

    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(8, noise_dim, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Optimizers for both G and D
    lr = 0.0002
    beta1 = 0.5
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

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

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data["X"].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
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
            label.fill_(fake_label)
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
            if (iters % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(plot_batch(fake))

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
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(i, animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())


if __name__ == "__main__":
    train_gan3d(img_dirs="D:Data/Beton/HPC/xyz-100-npy/", loadG="nets/netG", loadD="nets/netD",
                checkpoints=True, num_epochs=5)

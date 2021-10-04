import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] ='D:\\ffmpeg\\bin\\ffmpeg.exe'
from matplotlib.animation import FFMpegWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import Betondataset, plot_batch


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
            # nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=4,
                stride=1, padding=1, bias=False
            ),
            # nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=4,
                stride=1, padding=1, bias=False
            ),
            # nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv3_channels, out_channels=out_conv_channels, kernel_size=4,
                stride=1, padding=1, bias=False
            ),
            # nn.BatchNorm3d(out_conv_channels),
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


def train_net(load="", checkpoints=True, num_epochs=5):
    """
    Train a Classifier to generate more data

    :param load: if not "", load Classifier from here and continue training
    :param checkpoints: if state of the net should be saved after each epoch
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
    trainloader, testloader = Betondataset("semisynth-inf", binary_labels=True, batch_size=4, shuffle=True, num_workers=1)

    # batch = next(iter(trainloader))
    # print(batch["y"])
    # # plot_batch(batch["y"])
    # plot_batch(batch["X"])
    # plt.show()

    # Animate
    fig, anim_ax = plt.subplots(figsize=(8, 5))
    Writer = FFMpegWriter(fps=5)
    Writer.setup(fig, "cnn_progress.mp4", dpi=100)

    # Net
    # todo: smaller batch or more out_conv?
    net = Net(out_conv_channels=256).to(device)
    if load != '':
        try:
            net.load_state_dict(torch.load(load))
        except FileNotFoundError:
            print("No parameters loaded")
            pass

    # Loss / Optimizer
    # todo: weight FN more with pos_weight > 1
    # todo: use CrossEntropyLoss and extra category (e.g. unsure / nothing)
    pos_weight = 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    lr = 0.001
    weight_decay = 0.1
    beta1 = 0.9
    optimizer = optim.Adam(net.parameters(), betas=(beta1, 0.999), lr=lr, weight_decay=weight_decay)

    # Training loop
    losses = []
    iters = 0
    loss_mean = 0

    print("Starting Training Loop...")
    for epoch in range(1, num_epochs + 1):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data["X"].to(device), data["y"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Output training stats
            loss_mean += loss.item()
            losses.append(loss.item())
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f'
                      % (epoch, num_epochs, i, len(trainloader), loss_mean))
                loss_mean = 0

            iters += 1

        if checkpoints:
            # do checkpointing
            torch.save(net.state_dict(), '%s_epoch_%d' % (load or "netG", epoch))

        metrics(net, testloader, plot=epoch == num_epochs, anim=(Writer, anim_ax), criterion=criterion)

    Writer.finish()

    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.savefig("prog.png")
    plt.show()


def metrics(net, testloader, plot=True, anim=None, criterion=None):
    """
    Compute accuracy, recall, precision etc of a net on a test set

    :param anim: (Writer, ax): plot on ax and grab frame.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cutoff = 0.5
    pos_out = []

    neg_out = []
    tp, tn, fp, fn = 0, 0, 0, 0

    # Find 8 samples of FP and FN
    examples_fp = torch.zeros([8, 1, 100, 100, 100])
    examples_fn = torch.zeros([8, 1, 100, 100, 100])
    fp_idx = np.zeros([8])
    fn_idx = np.zeros([8])
    fp_n = 0
    fn_n = 0
    loss = 0

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            inputs, labels = batch["X"].to(device), batch["y"].to(device)
            outputs = torch.sigmoid(net(inputs))
            predicted = (outputs > cutoff).float().view(-1)

            if criterion is not None:
                loss += 1 / len(testloader) * criterion(outputs.view(-1), labels).mean().item()

            pos_out += outputs[labels == 1].view(-1).tolist()
            neg_out += outputs[labels == 0].view(-1).tolist()
            tp += ((predicted == 1.0) & (labels == 1.0)).sum().item()
            tn += ((predicted == 0.0) & (labels == 0.0)).sum().item()
            fp += ((predicted == 1.0) & (labels == 0.0)).sum().item()
            fn += ((predicted == 0.0) & (labels == 1.0)).sum().item()

            if plot:
                fp_ths = ((predicted == 1.0) & (labels == 0.0))
                fn_ths = ((predicted == 0.0) & (labels == 1.0))
                if fp_n < 8 and fp_ths.sum().item() > 0:
                    examples_fp[fp_n:min(8, fp_n + fp_ths.sum().item())] = inputs[fp_ths][0:8-fp_n]
                    fp_idx[fp_n:min(8, fp_n + fp_ths.sum().item())] = batch["id"][fp_ths][0:8-fp_n]
                    fp_n = min(8, fp_n + fp_ths.sum().item())
                if fn_n < 8 and ((predicted == 0.0) & (labels == 1.0)).sum().item() > 0:
                    examples_fn[fn_n:min(8, fn_n + fn_ths.sum().item())] = inputs[fn_ths][0:8-fn_n]
                    fn_idx[fn_n:min(8, fn_n + fn_ths.sum().item())] = batch["id"][fn_ths][0:8-fn_n]
                    fn_n = min(8, fn_n + fn_ths.sum().item())
    net.train()

    total = tp + tn + fp + fn
    recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    acc = 100 * (tp + tn) / total
    print('On the %d test images\nTP|TN|FP|FN: %d %d %d %d\nAccuracy: %.2f %%\nPrecision: %.2f %%\nRecall: %.2f %%' %
          (total, tp, tn, fp, fn, acc, precision, recall))
    if criterion is not None:
        print("Loss: %.2f" % loss)

    if anim is not None:
        bins = np.linspace(0, 1, 40)
        Writer, ax = anim
        ax.clear()
        ax.hist(pos_out, bins, alpha=0.5, label="pos")
        ax.hist(neg_out, bins, alpha=0.5, label="neg")
        ax.legend()
        Writer.grab_frame()

    if plot:
        print(fp_idx, fn_idx)
        plot_batch(examples_fn)
        plot_batch(examples_fp)
        plt.show()

        plt.figure()
        bins = np.linspace(0, 1, 40)
        plt.hist(pos_out, bins, alpha=0.5, label="pos")
        plt.hist(neg_out, bins, alpha=0.5, label="neg")
        plt.legend()
        plt.show()


def inspect_net(path):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net(out_conv_channels=256).to(device)
    net.load_state_dict(torch.load(path, map_location=device))

    trainloader, testloader = Betondataset("semisynth", binary_labels=True, batch_size=4, shuffle=True, num_workers=1)

    metrics(net, testloader)


if __name__ == "__main__":
    train_net(load="nets/netcnn_s", checkpoints=True, num_epochs=20)
    # inspect_net("nets/netcnn_s_epoch_5_epoch_19")

    """
    0.1 0.8 363 222 120 15
    """

import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib.animation import FFMpegWriter
from torchvision.transforms import Resize


from data import Betondataset, BetonImg, plot_batch, Resize_3d, Rotate_flip_3d
from paths import *
# from cnn_3d import Net
from models import LegNet1


"""
Fit shift and scale of a test dataset to a net by inspection
"""


def find_normalization(net, data, labels=None, num_iters=200, pos_weight=1.0, start_lr=1, preload_gpu=False, init=(0.0, 1.0)):
    """
    Find good normalization by minimizing loss w.r.t. shift/scale on a small sample

    :param net: PyTorch module
    :param data: PyTorch dataloader of inputs
    :param labels: list of labels, if None data is assumed to be labeled
    :param num_iters, iterations of Adam, default 200
    :param pos_weight, if > 1 priorities cracks
    """
    # Only train scale and shift
    for param in net.parameters():
        param.requires_grad = False
    dtype = next(net.parameters()).dtype  # Use dtype of net (i.e. float32)
    shift = Variable(torch.tensor(init[0]), requires_grad=True)
    scale = Variable(torch.tensor(init[1]), requires_grad=True)

    # preload all data (also to gpu), there should only be very few batches anyway
    batches = [batch["X"].to(dtype) for batch in data]
    if labels is not None:
        pos = [0] + list(np.cumsum([batch.shape[0] for batch in batches]))
        labels = [torch.tensor(labels)[pos[j]:pos[j + 1]].to(dtype) for j in range(len(pos) - 1)]
    else:
        labels = [batch["y"].to(dtype) for batch in data]
    if preload_gpu:
        batches = [batch.to(device) for batch in batches]
        labels = [label.to(device) for label in labels]

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(dtype).to(device))
    optimizer = optim.Adam([shift, scale], betas=(0.9, 0.999), lr=start_lr)
    # devide by 10 every iters/5, thus 1 -> 1e-5
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (5 / num_iters))

    losses = []
    for i in range(1, num_iters+1):
        if i % 5 == 0:
            print("%.1f %% - %.2f, (%.4f, %.4f)" % (100 * i / num_iters, np.mean(losses[-5:]), shift.item(), scale.item()))
        j = 0
        cur_loss = 0
        optimizer.zero_grad()
        for batch, label in zip(batches, labels):
            size = batch.shape[0]
            outputs = net((batch.to(device) - shift) / scale).view(-1)
            loss = criterion(outputs, label.to(device))
            loss.backward()
            cur_loss += loss.item() * size
            j += size
        optimizer.step()
        scheduler.step()
        losses += [cur_loss / 8]

    plt.plot(losses)
    plt.show()
    print(shift.item(), scale.item())
    return shift.item(), scale.item()


if __name__ == "__main__":
    # Seed
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    TUBE = [TUBE_PATH, 8, 0.58, 8.24,
            [(13, 3, 14), (13, 4, 14), (8, 20, 14), (8, 22, 14),
             (6, 6, 0), (6, 7, 0), (10, 5, 0), (10, 6, 0)],
            [1, 1, 1, 0, 0, 0, 1, 1]]
    HPC_16 = [HPC_16_PATH, 16, 0.098, 10.923,
              [(2, 3, 0), (2, 4, 0), (1, 8, 0), (1, 9, 0),
               (2, 9, 5), (4, 7, 5), (4, 4, 4), (4, 5, 4)],
              [1, 1, 1, 1, 1, 0, 0, 0]]
    path, bits, val_shift, val_scale, idxs, labels = TUBE  # HPC_16
    val_set = BetonImg(path, max_val=2 ** bits - 1)
    val = val_set.dataloader(batch_size=4, idxs=idxs, shuffle=False)

    # bits, val_shift, val_scale, labels = 8, 0, 1, None
    # val = Betondataset("real-val", batch_size=4, shuffle=False, test=0, norm=(0, 2**bits - 1))
    # synth = Betondataset("semisynth-inf", batch_size=4, shuffle=False, test=0, sampler=iter(list(range(8))))

    # Normalization on training
    train_shift = 0.11
    train_scale = 1

    # Net
    # net = Net(layers=1, dropout=0.0).to(device)
    net = LegNet1(layers=1).to(device)
    net.load_state_dict(torch.load("checkpoints/shift_0_11/netcnn_l1p_epoch_5.cp", map_location=device))
    net.eval()

    # nxt_synth = next(iter(synth))["X"]
    # nxt_val = next(iter(val))["X"]
    # plot_batch(nxt_val)
    # plt.show()

    # Tube: (0.3995, 0.7895) [w=0.67(1)], (0.4318, 1.1203) [w=1(1.7)], 0.5452, 2.5132 [w=1.5(2.5)], 0.4674, 1.3750 [w=2(3.4)], (0.4805, 0.8895) [w=1000]
    # HPC-16: 0.098, 10.92 [w=1(1.7)], 0.098, 10.923 [w=1.5(2.5)]
    # Shai's real: (0.0569, 1.1049) [w=1], 0.082, 1.33 [w=1.5], (0.0644, 1.0428) [w=2], (0.0980, 1.1425) [w=5]
    # Shai's real: (14.5267, 281.7497) [w=1]
    # Synth: 0, 1

    # find_normalization(net, synth, labels, num_iters=200, pos_weight=1.5)
    find_normalization(net, val, labels, num_iters=500, pos_weight=1, start_lr=0.01, init=(0.4, 0.86))

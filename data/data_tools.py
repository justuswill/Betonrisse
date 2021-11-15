import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.utils import make_grid
import torch.nn.functional as F


def plot_batch(x, acc=2, ax=None, title=None, scale=0):
    """
    Accumulate a dimension into batch dimension and plot a sample of 8 pictures

    :param acc:   which dimension to accumulate, showing different slices: 2 (xy), 1 (xz) or 0 (yz)
    :param scale: modify brightness. if > 1 brighter else darker, only works for data in range [0,1].
                  disable with scale = 0
    :return image that can be shown with imshow later
    """
    if ax is None:
        plt.figure(figsize=(8, 8))
    else:
        plt.sca(ax)

    sz = x.shape[-1]
    plt.axis("off")
    if title is None:
        slice = "". join([d for i, d in enumerate(["x", "y", "z"]) if i != acc])
        title = "Training Images (%s)" % slice
    plt.title(title)
    img = np.transpose(make_grid(torch.reshape(torch.transpose(torch.transpose(
            x, 2, 2 + acc)[:, :, 0:sz:(sz // 7), :, :], 1, 2), (-1, 1, sz, sz)),
            padding=2, normalize=scale == 0).cpu(), (1, 2, 0))
    if scale != 0:
        img = np.clip(scale * img, 0, 1)
    plt.imshow(img)
    return img


class ToTensor:
    def __call__(self, t):
        return torch.from_numpy(t)


class normalize:
    # HPC/riss: 33.24, 6.69
    # HPC:      32.69, 4.98
    # Semi:     30,    6.5
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, t):
        return (t - self.mean) / self.std


class normalize_each:
    """ Normalize with per-image mean and var """
    def __call__(self, t):
        return (t - t.mean()) / t.std()


class resize:
    def __init__(self, sz):
        self.sz = sz

    def __call__(self, t):
        return torch.squeeze(F.interpolate(t[None, :], size=self.sz, mode="trilinear", align_corners=False), 0)


class randomCrop:
    def __init__(self, sz):
        self.sz = sz

    def __call__(self, t):
        s = np.random.choice(t.shape[-1] - self.sz, size=3)
        return t[:, s[0]:s[0] + self.sz, s[1]:s[1] + self.sz, s[2]:s[2] + self.sz]


class random_rotate_flip_xy:
    def __call__(self, t):
        # flip
        if np.random.choice([True, False]):
            t = torch.flip(t, [1])
        # rotate xy planes
        rot = np.random.choice(4)
        t = torch.rot90(t, rot, (1, 2))
        return t


class random_rotate_flip_3d:
    def __init__(self, cache=True):
        """
        Randomly rotate or mirror a cube to any of the 48 possible orientations

        cache - save the last rotation to apply it again (only exactly once more)
        """
        self.cache = cache
        self.perm = None
        self.flips = None

    def __call__(self, t):
        # permute axis
        if self.perm is None:
            perm = [0] + list(np.random.permutation(3) + 1)
            if self.cache:
                self.perm = perm
        else:
            perm = self.perm
            self.perm = None
        t = t.permute(*perm)

        # flip
        if self.flips is None:
            flips = [np.random.choice([True, False]) for _ in range(3)]
            if self.cache:
                self.flips = flips
        else:
            flips = self.flips
            self.flips = None
        for i in range(3):
            if flips[i]:
                t = torch.flip(t, [i+1])
        return t


def mean_std(data):
    """
    compute mean and std of a dataset
    """
    dataloader = DataLoader(data, batch_size=8, shuffle=False, num_workers=2)

    # Compute mean and std
    mean = 0
    for i, batch in enumerate(dataloader):
        batch_mean = batch["X"].mean()
        mean += batch_mean * batch["X"].size(0)
        if i % 50:
            print("%.2f %% - %.2f" % (100 * i / len(dataloader), batch_mean))
    mean = mean / len(data)
    print(mean)

    var = 0
    for i, batch in enumerate(dataloader):
        batch_var = torch.mean((batch["X"] - mean) ** 2)
        var += batch_var * batch["X"].size(0)
        if i % 50:
            print("%.2f %% - %.2f" % (100 * i / len(dataloader), batch_var))
    var = var / len(data)
    std = np.sqrt(var)
    print("mean: %.2f \\ std: %.2f" % (mean, std))


def data_max(data):
    """
    compute max of a dataset
    """
    dataloader = DataLoader(data, batch_size=8, shuffle=False, num_workers=2)

    # Compute mean and std
    max = 0
    for i, batch in enumerate(dataloader):
        batch_max = batch["X"].max().item()
        if batch_max > max:
            max = batch_max
        if i % 50:
            print("%.2f %% - %.2f" % (100 * i / len(dataloader), max))
    print(max)


def data_hist(data, mult=1, ax=None):
    """
    compute max of a dataset
    """
    dataloader = DataLoader(data, batch_size=8, shuffle=False, num_workers=2)

    bins = np.arange(0, 256)
    hist = np.zeros(bins.shape)

    # Compute mean and std
    for i, batch in enumerate(dataloader):
        array = mult * np.array(batch["X"])
        hist += np.bincount(array.reshape(-1).astype(np.int), minlength=256)
        if i % 50:
            print("%.2f %%" % (100 * i / len(dataloader)))

    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(bins, hist, width=1)
    ax.set_yscale("log")


def train_test_dataloader(data, test_split=0.2, shuffle=False, **kwargs):
    """
    Create a train/test split of the dataset and create two dataloaders
    """

    # Creating data indices for training and test splits:
    size = len(data)
    indices = list(range(size))
    split = int(np.floor(test_split * size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(data, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(data, sampler=test_sampler, **kwargs)

    return train_loader, test_loader

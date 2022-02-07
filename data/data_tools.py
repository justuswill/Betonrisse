import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.utils import make_grid


"""
functions for plotting and analyzing datasets
"""


def plot_batch(x, acc=2, ax=None, title=None, scale=None):
    """
    Plot a batch by plotting cross-sections at different depths of the 3D Image.
    Each row is one sample, each column one depth.

    :param x: batch of shape B x C x H X W x D
    :param acc:   which dimension to accumulate per row, showing different slices: 2 (xy), 1 (xz) or 0 (yz)
    :param scale: modify brightness. if > 1 brighter else darker, only works for data in range [0,1], optional
                  if not given normalize to min and max value found in batch
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
            padding=2, normalize=scale is None).cpu(), (1, 2, 0))
    if scale is not None:
        img = np.clip(scale * img, 0, 1)
    plt.imshow(img)
    return img


def mean_std(data, workers=2, batch_size=8):
    """
    compute mean and std of a dataset
    """
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=workers)

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

    # Compute max
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
   plot histogram of a dataset
    """
    dataloader = DataLoader(data, batch_size=8, shuffle=False, num_workers=2)

    bins = np.arange(0, 256)
    hist = np.zeros(bins.shape)

    # Compute hist
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

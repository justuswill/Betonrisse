import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.utils import make_grid


def plot_batch(x, acc=2):
    """
    Accumulate a dimension into batch dimension and plot a sample of 8 pictures

    :param acc: which dimension to accumulate, showing different slices: 2 (xy), 1 (xz) or 0 (yz)
    :return image that can be shown with imshow later
    """
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    slice = "". join([d for i, d in enumerate(["x", "y", "z"]) if i != acc])
    plt.title("Training Images (%s)" % slice)
    img = np.transpose(make_grid(torch.reshape(torch.transpose(torch.transpose(
            x, 2, 2 + acc)[:, :, 0:80:10, :, :], 1, 2), (-1, 1, 100, 100)),
            padding=2, normalize=True).cpu(), (1, 2, 0))
    plt.imshow(img)
    return img


class ToTensor:
    def __call__(self, t):
        return torch.from_numpy(t)


class normalize:
    # HPC/riss: 33.24, 6.69
    # HPC:      32.69, 4.98
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, t):
        return (t - self.mean) / self.std


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


def train_test_dataloader(data, test_split=0.2, shuffle=True, **kwargs):
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

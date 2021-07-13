import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class Betondata(Dataset):
    def __init__(self, img_dirs, preload=False, transform=None):
        """
        Load npy from (multiple) locations

        :param img_dirs: path (or list of paths) to image npy files
        :param transform: apply some transforms like cropping, rotating, etc on input image
        :param preload; if npy should be loaded in memory

        todo: .npy is loaded (was saved) as int
        """
        if not isinstance(img_dirs, list):
            img_dirs = [img_dirs]

        self.img_names = [dir + f for dir in img_dirs for f in os.listdir(dir)]
        self.img_names = sorted(filter(lambda k: k.endswith("npy"), self.img_names))
        self.preload = preload
        self.transform = transform

        if self.preload:
            self.imgs = [np.load(img).astype(np.float32) for img in self.img_names]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.preload:
            sample = self.imgs[idx]
        else:
            sample = np.load(self.img_names[idx]).astype(np.float32)
        # Grayscale, so only one channel
        sample = sample[np.newaxis, :, :]
        if self.transform is not None:
            sample = self.transform(sample)
        return {"X": sample}


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
    plt.pause(0.00001)
    return img


class normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, t):
        return (t - self.mean) / self.std


if __name__ == "__main__":
    data = Betondata(img_dirs="D:Data/Beton/HPC/xyz-100-npy/",
                     transform=transforms.Lambda(normalize(32.69, 4.98)))
    dataloader = DataLoader(data, batch_size=8, shuffle=True, num_workers=2)

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
        batch_var = torch.mean((batch["X"] - mean)**2)
        var += batch_var * batch["X"].size(0)
        if i % 50:
            print("%.2f %% - %.2f" % (100 * i / len(dataloader), batch_var))
    var = var / len(data)
    std = np.sqrt(var)
    print("mean: %.2f \\ std: %.2f" % (mean, std))

    batch = next(iter(dataloader))["X"]

    plot_batch(batch, 2)
    plot_batch(batch, 1)
    plot_batch(batch, 0)
    plt.show()

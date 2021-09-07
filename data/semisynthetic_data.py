import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms

from .data import Betondata
from .synthetic_data import Synthdata
from .data_tools import ToTensor, randomCrop, random_rotate_flip_3d


class SemiSynthdata(Dataset):
    def __init__(self, n=128, size=1000, empty=True, binary_labels=False, num_cracks=1,
                 transform=None, data_transform=None, **kwargs):
        """
        Generate 3d images of cracks with brownian surfaces and optional fractal noise

        :param n: images are of size dim x dim x dim
        :param size: number of samples
        :param empty: if samples should include pictures without cracks (e.g. only noise)
        :param binary_labels: if set true, labels are 1 if its an image of a crack, 0 else.
                              if set false, labels are the same size as the picture with 1 where the crack is.
        :param transform: apply transforms to data and labels
        :param data_transform: apply transforms only to data
        :param kwargs: args for the brownian surface (i.e. crack simulation)
        """

        bg = Betondata(img_dirs="D:/Data/Beton/Semi-Synth/bg-npy-256/",
                       transform=transforms.Compose([
                            transforms.Lambda(ToTensor()),
                            transforms.Lambda(randomCrop(100)),
                            transforms.Lambda(random_rotate_flip_3d())
                       ]))
        noise = DataLoader(bg, batch_size=1, shuffle=True, num_workers=0)

        synth = Synthdata(n=n, size=size, noise=False, cached=False, binary_labels=binary_labels, empty=empty, **kwargs,
                          transform=transforms.Lambda(ToTensor()))

        self.n = n
        self.size = size
        self.noise = noise
        self.noise_iter = iter(noise)
        self.synth = synth
        self.num_cracks = num_cracks
        self.transform = transform
        self.data_transform = data_transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self.synth[idx]["X"]
        for _ in range(self.num_cracks - 1):
            more_sample = self.synth[idx]["X"]
            sample = torch.clamp(sample * more_sample, max=1)
        label = 1 - sample
        try:
            noise = next(self.noise_iter)
        except StopIteration:
            self.noise_iter = iter(self.noise)
            noise = next(self.noise_iter)
        sample = torch.clamp(torch.squeeze(noise["X"], 0) / 255 * sample, max=1)

        if self.transform is not None:
            sample = self.transform(sample)
            label = self.transform(label)
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return {"X": sample, "y": label}


def create_semisynthetic(dest_input, dest_label, size=1000, width=3, num_cracks=1):
    data = SemiSynthdata(n=100, size=size, empty=True, width=width, num_cracks=1)
    for i, x in enumerate(data):
        if i >= size:
            break
        print(i)
        np.save(dest_input + "w%d-%d-%d.npy" % (width, num_cracks, i), x["X"])
        np.save(dest_label + "w%d-%d-%d.npy" % (width, num_cracks, i), x["y"])

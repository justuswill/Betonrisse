import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms

from .data import Betondata
from .synthetic_data import Synthdata
from .data_tools import ToTensor, randomCrop, random_rotate_flip_3d


class Datasetiter():
    def __init__(self, data):
        self.data = data

    def __next__(self):
        return self.data[np.random.choice(len(self.data))]


class SemiSynthdata(Dataset):
    def __init__(self, n=128, size=1000, binary_labels=False, num_cracks=1, random_scale=False, confidence=1,
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

        # max: 217, mean: 30.47, std: 5.91
        bg = Betondata(img_dirs="D:/Data/Beton/Semi-Synth/bg-npy-256/",
                       transform=transforms.Compose([
                            transforms.Lambda(ToTensor()),
                            transforms.Lambda(randomCrop(n)),
                            transforms.Lambda(random_rotate_flip_3d(cache=not binary_labels))
                       ]))

        synth = Synthdata(n=n, size=size, noise=False, empty=False, cached=False, binary_labels=True, **kwargs,
                          transform=transforms.Compose([
                              transforms.Lambda(ToTensor()),
                              transforms.Lambda(random_rotate_flip_3d(cache=not binary_labels))
                          ]))

        self.n = n
        self.size = size
        self.random_scale = random_scale
        self.noise_iter = Datasetiter(bg)
        self.synth = synth
        self.num_cracks = num_cracks if hasattr(num_cracks, "__getitem__") else [num_cracks]
        self.binary_labels = binary_labels
        self.confidence = confidence
        self.transform = transform
        self.data_transform = data_transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = torch.ones([1, self.n, self.n, self.n])

        ths_num_cracks = np.random.choice(self.num_cracks)
        for _ in range(ths_num_cracks):
            more_sample = self.synth[idx]["X"]
            sample = torch.clamp(sample * more_sample, max=1)

        if self.binary_labels:
            # set to 0:1-conf, 1:conf
            label = np.float32((1 - self.confidence) + (ths_num_cracks > 0) * (2 * self.confidence - 1))
        else:
            label = 1 - sample

        air_mean = 12
        noise_shift = 0
        air_scale = 2
        noise_scale = 1
        if self.random_scale:
            air_mean = np.random.normal(air_mean, 1.5)
            noise_shift = np.random.normal(noise_shift, 1)
            air_scale = np.random.normal(air_scale, 0.5)
            noise_scale = np.random.normal(noise_scale, 0.2)

        noise = noise_shift + noise_scale * torch.squeeze(next(self.noise_iter)["X"], 0)
        air = air_mean + air_scale * torch.randn(sample.shape)
        sample = torch.clamp((noise * sample + air * (1 - sample)) / 255, min=0, max=1)

        if self.transform is not None:
            sample = self.transform(sample)
            if not self.binary_labels:
                label = self.transform(label)
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return {"X": sample, "y": label, "id": 0}


def create_semisynthetic(dest_input, dest_label, size=1000, width=3, num_cracks=1):
    data = SemiSynthdata(n=100, size=size, empty=True, width=width, num_cracks=1)
    for i, x in enumerate(data):
        if i >= size:
            break
        print(i)
        np.save(dest_input + "w%d-%d-%d.npy" % (width, num_cracks, i), x["X"])
        np.save(dest_label + "w%d-%d-%d.npy" % (width, num_cracks, i), x["y"])

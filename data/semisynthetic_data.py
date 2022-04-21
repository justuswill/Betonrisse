import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms

from data.data_transforms import ToTensor, RandomCrop, Random_rotate_flip_3d
from data.synthetic_data import Synthdata
from data.real_data import Betondata
from paths import *


"""
Semisynthetic data combining cracks (simulated as Brownian surface) with real background images.
"""


class Datasetiter():
    def __init__(self, data):
        self.data = data

    def __next__(self):
        return self.data[np.random.choice(len(self.data))]


class SemiSynthdata(Dataset):
    def __init__(self, n=128, size=1000, binary_labels=False, num_cracks=1, random_scale=False, air=True,
                 confidence=1, corruption=0, corruption_mean=24.5, offset=0,
                 transform=None, data_transform=None, **kwargs):
        """
        Generate 3d images of cracks with brownian surfaces

        :param n: images are of size dim x dim x dim
        :param size: number of samples per epoch (doesn't really matter, no repetitions anyway)
        :param binary_labels: if set true, labels are 1 if its an image of a crack, 0 else.
                              if set false, labels are the same size as the picture with 1 where the crack is.
        :param num_cracks: list of possible number of cracks
        :param air: ifd true set air/crack values to a dark but nonzero gaussian noise
        :param random_scale: if True mean and scale of crack and background are randomly varied
        :param confidence: set labels of cracks to confidence (default 1)
                           and labels of non-cracks to 1 - confidence (default 0)

        :param corruption: when interpolating between air (gaussian noise) and bg pixels
                           fix a minimal contribution of the gaussian noise, also noising the bg
        :param corruption_mean: correct the mean of the gaussian noise in regions of less air to be this value

        :param transform: apply transforms to data and labels
        :param data_transform: apply transforms only to data, applied after transform
        :param kwargs: args for the brownian surface (i.e. crack simulation)
        """

        # max: 217, mean: 30.47, std: 5.91
        bg = Betondata(img_dirs=BG_PATH,
                       transform=transforms.Compose([
                            transforms.Lambda(ToTensor()),
                            transforms.Lambda(RandomCrop(n)),
                            transforms.Lambda(Random_rotate_flip_3d(cache=not binary_labels))
                       ]))

        if offset > 0:
            trn = transforms.Compose([transforms.Lambda(ToTensor()),
                                      transforms.Lambda(RandomCrop(n)),
                                      transforms.Lambda(Random_rotate_flip_3d(cache=not binary_labels))])
        else:
            trn = transforms.Compose([transforms.Lambda(ToTensor()),
                                      transforms.Lambda(Random_rotate_flip_3d(cache=not binary_labels))])

        synth = Synthdata(n=n + offset, size=size, noise=False, empty=False, cached=False, binary_labels=True, **kwargs,
                          transform=trn)

        self.n = n
        self.size = size
        self.air = air
        self.corruption = corruption
        self.corruption_mean = corruption_mean
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
        # Cracks
        sample = torch.ones([1, self.n, self.n, self.n])
        ths_num_cracks = np.random.choice(self.num_cracks)
        for _ in range(ths_num_cracks):
            more_sample = self.synth[idx]["X"]
            # print(torch.sum(more_sample == 0) / self.n ** 3 * 100)
            while torch.all(more_sample != 0):
                # print("redo")
                more_sample = self.synth[idx]["X"]
            sample = torch.clamp(sample * more_sample, max=1)

        if self.binary_labels:
            # set to 0 -> 1-conf, 1 -> conf
            label = np.float32((1 - self.confidence) + (ths_num_cracks > 0) * (2 * self.confidence - 1))
        else:
            label = 1 - sample

        noise_shift = 0
        noise_scale = 1
        if self.air:
            # by visual inspection of bg (peak at 12.5, extend 9-17)
            air_mean = 12.5
            air_scale = 2.5
        else:
            air_mean = 0
            air_scale = 0
        if self.random_scale:
            air_mean = np.random.normal(air_mean, 2)
            noise_shift = np.random.normal(noise_shift, 1)

        # Combine noise and air (crack) - weighted by sample
        noise = noise_shift + noise_scale * torch.squeeze(next(self.noise_iter)["X"], 0)
        air = air_mean + air_scale * torch.randn(sample.shape)
        if self.corruption > 0:
            # observe a noisy bg
            sample2 = torch.clamp(sample, min=0, max=1 - self.corruption)
            sample = torch.clamp((noise * sample2 + air * (1 - sample2) + (self.corruption_mean - air_mean) * (1 - sample)) / 255, min=0, max=1)
        else:
            sample = torch.clamp((noise * sample + air * (1 - sample)) / 255, min=0, max=1)

        if self.transform is not None:
            sample = self.transform(sample)
            if not self.binary_labels:
                label = self.transform(label)
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return {"X": sample, "y": label, "id": 0}


def create_semisynthetic(dest_input, dest_label, size=1000, width=3, num_cracks=1):
    data = SemiSynthdata(n=100, size=size, width=width, num_cracks=[0, 1])
    for i, x in enumerate(data):
        if i >= size:
            break
        print(i)
        np.save(dest_input + "w%d-%d-%d.npy" % (width, num_cracks, i), x["X"])
        np.save(dest_label + "w%d-%d-%d.npy" % (width, num_cracks, i), x["y"])

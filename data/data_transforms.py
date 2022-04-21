import itertools
import numpy as np

import torch
import torch.nn.functional as F


"""
Transformations on 3D Images
"""


class ToTensor:
    def __call__(self, t):
        return torch.from_numpy(t)


class Normalize:
    """
    Normalize tensors from a dataset with given mean and std.
    If shift is given, instead subtract after scaling (yields the same result for shift=mean/std
    """
    # HPC/riss: 33.24, 6.69
    # HPC:      32.69, 4.98
    # Semi:     30,    6.5
    def __init__(self, mean, std, shift=None):
        self.mean = mean if shift is None else 0
        self.std = std
        self.shift = shift

    def __call__(self, t):
        t -= self.mean
        t /= self.std
        if self.shift is not None:
            t -= self.shift
        return t


class Normalize_each:
    """ Normalize with per-image mean and var """
    def __call__(self, t):
        return (t - t.mean()) / t.std()


class Normalize_min_max:
    """ Normalize with per-image mean and var """
    def __call__(self, t):
        min = np.percentile(t, 0.1)
        max = np.percentile(t, 99.9)
        return (t - min) / (max - min)


class Normalize_median:
    """ Normalize with per-image mean and var """
    def __call__(self, t):
        return t - t.median()


class Resize:
    """ ? Resize tensors to new 3D shape by interpolation """
    def __init__(self, sz):
        self.sz = sz

    def __call__(self, t):
        """
        C x H x W x D -> C x sz[0] x sz[1] x sz[2]
        """
        return torch.squeeze(F.interpolate(t[None, :], size=self.sz, mode="trilinear", align_corners=False), 0)


class Resize_3d:
    """ Resize tensors to new 3D shape by interpolation"""
    def __init__(self, sz):
        self.sz = sz

    def __call__(self, t):
        """
        B x C x H x W x D -> B x C x sz[0] x sz[1] x sz[2]
        """

        meshz, meshy, meshx = torch.meshgrid([torch.linspace(-1, 1, self.sz[i]) for i in range(3)])
        grid = torch.stack((meshx, meshy, meshz), 3)
        grid = grid.unsqueeze(0).repeat(t.shape[0], 1, 1, 1, 1)

        return F.grid_sample(t, grid.to(t.dtype), align_corners=True)


class RandomCrop:
    """ randomly crop to a fixed size """
    def __init__(self, sz):
        self.sz = sz

    def __call__(self, t):
        """
        C x H x W x D -> C x sz[0] x sz[1] x sz[2]
        """
        s = np.random.choice(t.shape[-1] - self.sz, size=3)
        return t[:, s[0]:s[0] + self.sz, s[1]:s[1] + self.sz, s[2]:s[2] + self.sz]


class Random_rotate_flip_xy:
    """ randomly rotate and/or mirror in the xy plane, 8 possible orientations """
    def __call__(self, t):
        # flip
        if np.random.choice([True, False]):
            t = torch.flip(t, [1])
        # rotate xy planes
        rot = np.random.choice(4)
        t = torch.rot90(t, rot, (1, 2))
        return t


class Random_rotate_flip_3d:
    """ randomly rotate and/or mirror in all planes, 48 possible orientations """

    def __init__(self, cache=False):
        """
        :param cache: save the last rotation to apply it again (only exactly once more),
                      useful for also applying it to a label
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

        # flip
        if self.flips is None:
            flips = [np.random.choice([True, False]) for _ in range(3)]
            if self.cache:
                self.flips = flips
        else:
            flips = self.flips
            self.flips = None

        t = t.permute(*perm)
        for i in range(3):
            if flips[i]:
                t = torch.flip(t, [i+1])
        return t


class Rotate_flip_3d:
    """ deterministically rotate and/or mirror in all planes, 48 possible orientations """
    def __init__(self):
        self.perms = list(map(list, itertools.permutations([2, 3, 4])))

    def __call__(self, t, idx):
        """
        :param t: PyTorch Tensor B x C x H x W x D
        """
        fx = idx % 2
        fy = (idx // 2) % 2
        fz = ((idx // 2) // 2) % 2
        pm = ((idx // 2) // 2) // 2

        # rotate
        perm = [0, 1] + self.perms[pm]
        t = t.permute(*perm)

        # flip
        for i, fl in enumerate([fx, fy, fz]):
            if fl:
                t = torch.flip(t, [i+2])
        return t

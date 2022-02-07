import numpy as np

from torch.utils.data import Dataset

from .brownian_surface import generate_crack
from .noise import generate_fractal_noise_3d


"""
Synthetic data containing cracks (simulated as Brownian surface) and optional fractal/perlin noise.
"""


class Synthdata(Dataset):
    def __init__(self, n=128, size=1000, noise=True, snr=2, empty=True, cached=True, binary_labels=False,
                 transform=None, data_transform=None, **kwargs):
        """
        Generate 3d images of cracks with brownian surfaces and optional fractal noise

        :param n: images are of size dim x dim x dim
        :param size: number of samples
        :param noise: if fractal noise should be added
        :param snr: signal to noise ratio - noise range is set to [0, 2/snr] with a mean of 1/snr
        :param empty: if samples should include pictures without cracks (e.g. only noise)
        :param cached: if samples at fixed index should be cached and reloaded or generated anew
        :param binary_labels: if set true, labels are 1 if its an image of a crack, 0 else.
                              if set false, labels are the same size as the picture with 1 where the crack is.
        :param transform: apply transforms to data and labels
        :param data_transform: apply transforms only to data, applied after transform
        :param kwargs: parameters for brownian surfaces or fractal noise generation
        """

        self.n = n
        self.size = size
        self.noise = noise
        self.noise_scale = 2/snr
        self.empty = empty
        self.cached = cached
        self.cache = dict()
        self.binary_labels = binary_labels
        self.transform = transform
        self.data_transform = data_transform
        self.brown_kwargs = {k: v for k, v in kwargs.items() if k in ["width", "H", "sigma"]}
        self.noise_kwargs = {k: v for k, v in kwargs.items() if k in ["res", "octaves", "persistence", "lacunarity"]}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.cached and idx in self.cache.keys():
            return self.cache[idx]

        # Generate crack
        sample = np.zeros((1, self.n, self.n, self.n), dtype=np.float32)
        empty = self.empty and np.random.choice([True, False])
        if not empty:
            sample += generate_crack(self.n, **self.brown_kwargs)

        # label
        if self.binary_labels:
            label = float(not empty)
        else:
            label = sample.copy()
            if self.transform is not None:
                label = self.transform(label)

        # Add noise
        if self.noise:
            noise = generate_fractal_noise_3d((self.n, self.n, self.n), **self.noise_kwargs)
            sample = np.minimum(1, sample + self.noise_scale * noise)

        # invert as 0 = low brightness = crack
        sample = 1 - sample

        if self.transform is not None:
            sample = self.transform(sample)
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return {"X": sample, "y": label}


def create_synthetic(dest_input, dest_label, size=5000):
    data = Synthdata(n=100, size=size, empty=True, noise=True)
    for i, x in enumerate(data):
        if i >= size:
            break
        print(i)
        np.save(dest_input + "%d.npy" % i, x["X"])
        np.save(dest_label + "%d.npy" % i, x["y"])

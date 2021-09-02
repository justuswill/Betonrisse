import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from data_tools import plot_batch, ToTensor, normalize, random_rotate_flip_3d


class Betondata(Dataset):
    def __init__(self, img_dirs, label_dirs=None, binary_labels=False, transform=None):
        """
        Load npy from (multiple) locations

        :param img_dirs: path (or list of paths) to image npy files
        :param img_dirs: path (or list of paths) to label npy files,
            if a file here has the same name as a file in img_dirs it is its label.
            pictures without matching label get the all-zero label.
        :param binary_labels: if set true, labels are 1 if its an image of a crack, 0 else.
                              if set false, labels are the same size as the picture with 1 where the crack is.
        :param transform: apply some transforms like cropping, rotating, etc on input image

        todo: .npy is loaded (was saved) as int
        """
        if not isinstance(img_dirs, list):
            img_dirs = [img_dirs]

        self.img_names = [dir + f for dir in img_dirs for f in os.listdir(dir)]
        self.img_names = list(filter(lambda k: k.endswith("npy"), self.img_names))
        self.img_names = sorted(self.img_names, key=lambda s: int(s.split("/")[-1][:-4]))
        self.transform = transform

        # find matching labels
        self.labels = label_dirs is not None
        self.binary_labels = binary_labels
        if self.labels:
            if not isinstance(label_dirs, list):
                label_dirs = [label_dirs]

            label_pool = {f: dir + f for dir in label_dirs for f in os.listdir(dir)}
            self.label_names = [label_pool.get(name.split("\\")[-1], None) for name in self.img_names]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        sample = np.load(self.img_names[idx]).astype(np.float32)
        # Grayscale, so only one channel
        sample = sample[np.newaxis, :, :]
        if self.transform is not None:
            sample = self.transform(sample)
        if not self.labels:
            return {"X": sample}
        else:
            if self.label_names[idx] is not None:
                label = np.load(self.label_names[idx])
            elif self.binary_labels:
                label = 0.0
            else:
                label = np.zeros(sample.shape, sample.dtype)
            return {"X": sample, "y": label}


if __name__ == "__main__":
    data = Betondata(img_dirs="D:Data/Beton/HPC/riss/",
                     transform=transforms.Compose([
                         transforms.Lambda(ToTensor()),
                         transforms.Lambda(normalize(33.24, 6.69)),
                         transforms.Lambda(random_rotate_flip_3d())
                     ]))
    dataloader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0)

    # data = Betondata(img_dirs="D:Data/Beton/HPC/xyz-100-npy/",
    #                  transform=transforms.Lambda(normalize(32.69, 4.98)))

    for batch in dataloader:
       batch = batch["X"]
       plot_batch(batch, 2)
       plt.show()

    # batch = next(iter(dataloader))["X"]
    # plot_batch(batch, 2)
    # plot_batch(batch, 1)
    # plot_batch(batch, 0)
    # plt.show()

import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch


class Betondata(Dataset):
    def __init__(self, img_dirs, preload=False, transform=None):
        """
        Load npy from (multiple) locations

        :param img_dirs: path (or list of paths) to image npy files
        :param transform: apply some transforms like cropping, rotating, etc on input image
        :param preload; if npy should be loaded in memory
        """
        if not isinstance(img_dirs, list):
            img_dirs = [img_dirs]

        self.img_names = [dir + f for dir in img_dirs for f in os.listdir(dir)]
        self.img_names = sorted(filter(lambda k: k.endswith("npy"), self.img_names))
        self.preload = preload
        self.transform = transform

        if self.preload:
            self.imgs = [np.load(img) for img in self.img_names]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.preload:
            sample = self.imgs[idx]
        else:
            sample = np.load(self.img_names[idx])
        return {"X": sample}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = Betondata(img_dirs="D:Data/Beton/Real-1/xyz-100-npy/")
    dataloader = DataLoader(data, batch_size=4, shuffle=True, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i)
        print(torch.max(batch["X"]))
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(batch["X"][0][:, :, 0], cmap="gray")
        ax[0, 1].imshow(batch["X"][1][:, :, 0], cmap="gray")
        ax[1, 0].imshow(batch["X"][2][:, :, 0], cmap="gray")
        ax[1, 1].imshow(batch["X"][3][:, :, 0], cmap="gray")
        plt.show()

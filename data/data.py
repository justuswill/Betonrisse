import os
import numpy as np

from torch.utils.data import Dataset


class Betondata(Dataset):
    def __init__(self, img_dirs, label_dirs=None, binary_labels=False, transform=None, data_transform=None):
        """
        Load npy from (multiple) locations

        :param img_dirs: path (or list of paths) to image npy files
        :param img_dirs: path (or list of paths) to label npy files,
            if a file here has the same name as a file in img_dirs it is its label.
            pictures without matching label get the all-zero label.
        :param binary_labels: if set true, labels are 1 if its an image of a crack, 0 else.
                              if set false, labels are the same size as the picture with 1 where the crack is.
        :param transform: apply transforms to data and labels
        :param data_transform: apply transforms only to data
        """
        if not isinstance(img_dirs, list):
            img_dirs = [img_dirs]

        self.img_names = [dir + f for dir in img_dirs for f in os.listdir(dir)]
        self.img_names = list(filter(lambda k: k.endswith("npy"), self.img_names))
        # self.img_names = sorted(self.img_names, key=lambda s: int(s.split("/")[-1][:-4]))
        self.transform = transform
        self.data_transform = data_transform

        # find matching labels
        self.labels = label_dirs is not None
        self.binary_labels = binary_labels
        if self.labels:
            if not isinstance(label_dirs, list):
                label_dirs = [label_dirs]

            label_pool = {f: dir + f for dir in label_dirs for f in os.listdir(dir)}
            self.label_names = [label_pool.get(name.replace("/", "\\").split("\\")[-1], None) for name in
                                self.img_names]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        sample = np.load(self.img_names[idx]).astype(np.float32)

        # Grayscale, so only one channel
        if len(sample.shape) < 4:
            sample = sample[np.newaxis, :, :, :]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        out = {"X": sample, "id": idx}

        if self.labels:
            if self.label_names[idx] is not None:
                label = np.load(self.label_names[idx]).astype(np.float32)
                if len(label.shape) < 4:
                    label = label[np.newaxis, :, :, :]
                if self.binary_labels:
                    label = float(np.any(label))
                elif self.transform is not None:
                    label = self.transform(label)
            # No label means no crack
            else:
                if self.binary_labels:
                    label = 0.0
                else:
                    label = np.zeros(sample.shape, np.float32)
                    if self.transform is not None:
                        label = self.transform(label)
            out["y"] = label

        return out

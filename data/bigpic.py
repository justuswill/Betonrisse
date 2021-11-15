import io
import time
import tarfile
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# hacky
from PIL import ImageFile

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.rcParams['animation.ffmpeg_path'] = 'D:\\ffmpeg\\bin\\ffmpeg.exe'


# Dataloaders for slices (e.g. xy) with depth n and at least <overlap> layers in common with the previous slice
class SliceLoader(Dataset):
    def __init__(self, n, overlap):
        self.n = n
        self.overlap = overlap

    def size(self):
        # number of layers
        pass

    def getlayer(self, idx):
        # return layer idx as np array
        pass

    def shape(self):
        return list(self.getlayer(0).shape) + [self.size()]

    def __len__(self):
        return np.ceil((self.size() - self.n) / (self.n - self.overlap)).astype(np.int) + 1

    def __getitem__(self, idx):
        start = idx * (self.n - self.overlap)
        end = start + self.n
        if end > self.size():
            start = self.size() - self.n
            end = self.size()
        slice = np.stack([self.getlayer(layer) for layer in range(start, end)], axis=-1)
        return {"X": slice[None, :, :, :], "idx": idx}

    def dataloader(self, **kwargs):
        return DataLoader(self, shuffle=False, **kwargs)


class TarLoader(SliceLoader):
    def __init__(self, img_path, *args):
        super().__init__(*args)
        self.tf = tarfile.open(img_path)
        self.img_names = list(filter(lambda k: k.endswith("jpg"), self.tf.getnames()))

    def size(self):
        return len(self.img_names)

    def getlayer(self, idx):
        img = self.tf.extractfile(self.img_names[idx])
        # convert to greyscale array
        img = Image.open(io.BytesIO(img.read())).convert("L")
        img = np.array(img).astype(np.float32)
        return img


class TifLoader(SliceLoader):
    def __init__(self, img_path, *args):
        super().__init__(*args)
        self.img_stack = Image.open(img_path)
        self.img_stack.filename = None

    def size(self):
        return self.img_stack.n_frames

    def getlayer(self, idx):
        self.img_stack.seek(idx)
        return np.array(self.img_stack).astype(np.float32)


class BetonImg:
    def __init__(self, img_path, n=100, overlap=25, load=None):
        """
        Load, split and analyze a picture

        :param img_path: path to image file, currently supports:
            .tar (dir of xy slices)
            .tif
        :param n: input size of classification/segmentation
        """
        if img_path.endswith(".tar"):
            loader = TarLoader
        elif img_path.endswith(".tif"):
            loader = TifLoader
        else:
            raise ValueError("file type not supported")

        self.n = n
        self.overlap = overlap
        self.load = load
        self.slices = loader(img_path, n, overlap)
        # -1 wip, 0, no crack, 1 crack
        self.ankers = [sorted(list(set(range(0, s - n, (self.n - self.overlap))) | {s - n})) for s in self.shape()]
        # Load net
        self.predictions = -np.ones([len(ank) for ank in self.ankers])
        if load is not None:
            try:
                self.predictions = np.load(load)
            except FileNotFoundError:
                print("No existing predictions loaded")
                pass

    def shape(self):
        return self.slices.shape()

    def predict(self, net, device, transform=None):
        # todo: batch_size
        # todo: don't load predicted slices
        # todo: more workers?
        if not np.any(self.predictions == -1):
            print("no predictions made")
            return

        cutoff = 0.5

        start_full = time.time()
        dur_eval = 0
        net.eval()
        with torch.no_grad():
            for iz, slice in enumerate(self.slices.dataloader(batch_size=1)):
                print("[%.2f %%]" % (100 * iz / len(self.slices)))
                if self.predictions[0, 0, slice["idx"]] != -1:
                    continue
                for ix, x in enumerate(self.ankers[0]):
                    for iy, y in enumerate(self.ankers[1]):
                        input = slice["X"][:, :, x:x + self.n, y:y + self.n, :]
                        if transform is not None:
                            input = transform(input)
                        start = time.time()
                        output = torch.sigmoid(net(input.to(device))).cpu()
                        self.predictions[ix, iy, slice["idx"]] = (output > cutoff).float().view(-1)
                        dur_eval += time.time() - start
                np.save(self.load, self.predictions)
        net.train()
        dur_full = time.time() - start_full
        print("Time: %g s, evaluation %g s (%g s / sample)" %
              (dur_full, dur_eval, dur_eval / self.predictions.size))

    def animate(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.axis("off")
        Writer = FFMpegWriter(fps=20)
        Writer.setup(fig, "big_pic.mp4", dpi=100)

        for iz, b in enumerate(self.slices.dataloader(batch_size=1)):
            print("[%.2f %%]" % (100 * iz / len(self.slices)))
            z = b["idx"]
            slice = b["X"][0, 0, :, :, :]

            for d in range(slice.shape[2]):
                ax.clear()
                im = slice[:, :, d]
                ax.imshow(im)
                for ix, x in enumerate(self.ankers[0]):
                    for iy, y in enumerate(self.ankers[1]):
                        Bbox.from_extents(xmin, 0, xmax, 1)
                Writer.grab_frame()
        Writer.finish()

import io
import time
import tarfile
from PIL import Image
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.colors as mc
from matplotlib.animation import FFMpegWriter
from abc import ABC, abstractmethod
from typing import Sized, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler
import torchvision.transforms as transforms

from data import normalize
from paths import *
plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

# hacky, but fixes image loading
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


"""
Data object to handle large 3D images.

Data is loaded/stored using the SliceLoader class, a PyTorch dataset containing slices.
Prediction, plotting, etc is done by the BetonImg class,
which is also a (very inefficient, only for debugging) Pytorch dataset containing all 3D Image chunks.
"""


# Dataloaders for slices (e.g. xy) with depth n and at least <overlap> layers in common with the previous slice
class SliceLoader(ABC, Dataset):
    """
    PyTorch dataset for slices of width <n> with overlap.
    """
    def __init__(self, n, overlap, transform=None):
        self.n = n
        self.overlap = overlap
        self.transform = transform

    @abstractmethod
    def size(self):
        # number of layers
        pass

    @abstractmethod
    def getlayer(self, idx):
        # return layer at idx as np array
        pass

    def shape(self):
        return list(self.getlayer(0).shape) + [self.size()]

    def __len__(self):
        return np.ceil((self.size() - self.n) / (self.n - self.overlap)).astype(int) + 1

    def __getitem__(self, idx, transform=None):
        start = idx * (self.n - self.overlap)
        end = start + self.n
        if end > self.size():
            start = self.size() - self.n
            end = self.size()
        slice = np.stack([self.getlayer(layer) for layer in range(start, end)], axis=-1)
        if self.transform is not None:
            slice = self.transform(slice)
        return {"X": slice[None, :, :, :], "idx": idx}

    def dataloader(self, **kwargs):
        return DataLoader(self, shuffle=False, **kwargs)


# Loading a .tar archive of .jpg / .png files (of xy slices)
class TarLoader(SliceLoader):
    def __init__(self, img_path, *args):
        super().__init__(*args)
        self.tf = tarfile.open(img_path)
        self.img_names = sorted(filter(lambda k: k.endswith("jpg") or k.endswith("png"), self.tf.getnames()))

    def size(self):
        return len(self.img_names)

    def getlayer(self, idx):
        img = self.tf.extractfile(self.img_names[idx])
        # convert to greyscale array
        img = Image.open(io.BytesIO(img.read())).convert("L")
        img = np.array(img).astype(np.float32)
        return img


# Loading a .tif/.tiff
class TifLoader(SliceLoader):
    def __init__(self, img_path, *args):
        super().__init__(*args)
        self.img_stack = Image.open(img_path)

    def size(self):
        return self.img_stack.n_frames

    def getlayer(self, idx):
        self.img_stack.seek(idx)
        return np.array(self.img_stack).astype(np.float32)


# Sampler for slices
class RangeSampler(Sampler):
    """
    Samples elements sequentially from a range, always in the same order,
    """
    data_source: Sized

    def __init__(self, data_source: Sized, start=0) -> None:
        self.data_source = data_source
        self.start = 0

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class BetonImg(Dataset):
    def __init__(self, img_path, n=100, overlap=25, max_val=255, batch_size=8, transform=None, load=None):
        """
        Load, split and analyze a large 3D image.
        As a Pytorch dataset it provides (very inefficiently) image chunks.

        :param img_path: path to image file, currently supports:
            .tar (dir of .jpg / .png of xy slices)
            .tif / .tiff
        :param n: input size of classification/segmentation as (n x n x n), default: 100
        :param overlap: number of shared pixels with neighboring image chunks, default: 25
        :param max_val: biggest pixel value possible in loaded image format, default: 2**8 (8 bit grayscale)
        :param batch_size: number of chunks in each batch for prediction
        :param transform: Callable for image transformation (called on images with pixel values in [0, 1]), optional
        :param load: path for loading of (partial) predictions, optional
        """
        if img_path.endswith(".tar"):
            loader = TarLoader
        elif img_path.endswith(".tif"):
            loader = TifLoader
        else:
            raise ValueError("file type not supported")

        self.n = n
        self.overlap = overlap
        self.max_val = max_val
        self.batch_size = batch_size
        self.transform = transform
        self.load = load
        self.slices = loader(img_path, n, overlap, transforms.Lambda(normalize(0, max_val)))
        # starting locations for each chunk
        self.anchors = [sorted(list(set(range(0, s - n, (self.n - self.overlap))) | {s - n})) for s in self.shape()]
        # -1: tbd, 0: no crack, 1: crack
        self.predictions = -1 * np.ones([len(ank) for ank in self.anchors])
        if load is not None:
            try:
                self.predictions = np.load(load)
            except FileNotFoundError:
                print("No existing predictions loaded")
                pass

    # shape in number of chunks (not number of pixels)
    def shape(self):
        return self.slices.shape()

    def __len__(self):
        return np.prod([len(ank) for ank in self.anchors])

    def __getitem__(self, idxs):
        if hasattr(idxs, "__getitem__"):
            ix, iy, iz = idxs
        else:
            iz = idxs // (len(self.anchors[0]) * len(self.anchors[1]))
            ixy = idxs % (len(self.anchors[0]) * len(self.anchors[1]))
            iy = ixy // len(self.anchors[0])
            ix = ixy % len(self.anchors[0])
        return {"X": self.slices[iz]["X"][:, self.anchors[0][ix]: self.anchors[0][ix] + self.n, self.anchors[1][iy]: self.anchors[1][iy] + self.n, :]}

    def dataloader(self, **kwargs):
        if "idxs" in kwargs.keys():
            return DataLoader(self, sampler=SubsetRandomSampler(kwargs.pop("idxs")), **kwargs)
        else:
            return DataLoader(self, **kwargs)

    def _predict_batch(self, net, device, batch, batch_ix, batch_iy, batch_iz):
        """
        Predict a batch of chunks

        :return: evaluation time
        """
        batch_output = []
        dur_eval = 0
        start = time.time()
        batch_output += [net(batch.to(device)).cpu().float().view(-1)]
        dur_eval += time.time() - start
        for out, ix, iy, iz in zip(batch_output, batch_ix, batch_iy, batch_iz):
            self.predictions[ix, iy, iz] = out
        return dur_eval

    def predict(self, net, device, transform=None):
        """
        Predict all chunks of the image.

        :param net: model (with loaded parameters)
        :param device: PyTorch device
        :param transform:
        :return:
        """
        # todo: more workers? -> complex, not yet necessary
        if not np.any(self.predictions == -1):
            print("no predictions made")
            return

        # Continue partial prediction
        idxs = np.any(self.predictions == -1, axis=(0, 1))
        dataloader = self.slices.dataloader(batch_size=1, sampler=iter(idxs))
        batch_input, batch_ix, batch_iy, batch_iz = [], [], [], []

        save_next_batch = False
        start_full = time.time()
        dur_eval = 0
        net.eval()
        with torch.no_grad():
            for iz, slice in enumerate(dataloader):
                print("[%.2f %%]" % (100 * iz / len(dataloader)))
                for ix, x in enumerate(self.anchors[0]):
                    for iy, y in enumerate(self.anchors[1]):
                        if self.predictions[ix, iy, slice["idx"]] != -1:
                            continue
                        input = slice["X"][:, :, x:x + self.n, y:y + self.n, :]
                        if self.transform is not None:
                            input = self.transform(input)

                        batch_input += [input]
                        batch_ix += [ix]
                        batch_iy += [iy]
                        batch_iz += [slice["idx"]]

                        if len(batch_input) == self.batch_size:
                            dur_eval += self._predict_batch(net, device, torch.cat(batch_input), batch_ix, batch_iy, batch_iz)
                            if save_next_batch:
                                np.save(self.load, self.predictions)
                                save_next_batch = False
                # checkpoint after each slice
                save_next_batch = True

            # final batch + checkpoint
            if len(batch_input) > 0:
                dur_eval += self._predict_batch(net, device, torch.cat(batch_input), batch_ix, batch_iy, batch_iz)
            np.save(self.load, self.predictions)

        net.train()
        dur_full = time.time() - start_full
        print("Time: %g s, evaluation %g s (%g s / sample)" %
              (dur_full, dur_eval, dur_eval / self.predictions.size))

    def _animate_layer(self, im, predictions, mode="alpha", ax=None):
        """
        animate one layer of the prediction.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 10))

        cutoff = 0.5

        # uncertainty is currently messy, because net doesn't give proper estimates
        # therefore we use net output directly, clipping to a max/min value found by inspection
        # values roughly between 5% and 95%
        # b = 3
        b = 500
        norm = mc.Normalize(vmin=-b, vmax=b, clip=True)

        ax.clear()
        ax.set_axis_off()
        ax.imshow(im, cmap='gray', vmin=0, vmax=1, origin="lower")
        for ix, x in enumerate(self.anchors[0]):
            for iy, y in enumerate(self.anchors[1]):
                pred = torch.tensor(predictions[ix, iy])

                if mode == "clas":
                    if torch.sigmoid(pred) > cutoff:
                        ax.add_patch(mpatch.Rectangle((y, x), self.n, self.n, fill=True, fc="red", alpha=0.3))
                elif mode == "cmap":
                    c = list(plt.cm.get_cmap("RdYlGn_r")(norm(pred)))
                    c[3] = 0.3 * c[3]
                    ax.add_patch(mpatch.Rectangle((y, x), self.n, self.n, fill=True, fc=c))
                elif mode == "alpha":
                    alpha = 0.3 * norm(pred)
                    ax.add_patch(mpatch.Rectangle((y, x), self.n, self.n, fill=True, fc="red", alpha=alpha))
                elif mode == "cmap-alpha":
                    # alpha = 2 * abs(norm(pred) - 0.5)
                    # alpha = 0.3 * alpha if alpha > 0.4 else 0
                    alpha = 0.3 * norm(pred)
                    c = list(plt.cm.get_cmap("RdYlGn_r")(norm(pred)))
                    c[3] = alpha
                    ax.add_patch(mpatch.Rectangle((y, x), self.n, self.n, fill=True, fc=c))
                elif mode == "none":
                    pass
                else:
                    raise ValueError("Mode not supported")

    def animate(self, mode="alpha", filename="big_pic.mp4"):
        """
        animate the prediction where the z-axis is time.

        :param mode: "clas": classification wrt. cutoff
                     "cmap": color wrt. output
                     "alpha": alpha wrt. output
                     "cmap-alpha": color/alpha wrt. output
                     "none": only img
        :param filename: where to save the animation
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        Writer = FFMpegWriter(fps=40)
        Writer.setup(fig, filename, dpi=100)

        for iz, b in enumerate(self.slices.dataloader(batch_size=1)):
            print("[%.2f %%]" % (100 * iz / len(self.slices)))
            z = b["idx"]
            slice = b["X"][0, 0, :, :, :]

            for d in range(slice.shape[2]):
                # skip start of slice when overlapping
                # todo: overlap on last slice may be bigger
                if iz > 0 and d < self.overlap:
                    continue
                # combine predictions by mean
                pred = self.predictions[:, :, z]
                if (d > self.n - self.overlap) and (z + 1 < self.predictions.shape[2]):
                    pred = np.mean([pred, self.predictions[:, :, z + 1]], axis=0)
                self._animate_layer(slice[:, :, d], pred, mode=mode, ax=ax)
                Writer.grab_frame()
        Writer.finish()

    def plot_layer(self, idx, mode="alpha"):
        """
        plot on layer of the prediction

        :param idx: idx of layer
        :param mode: see self.animate
        """
        pred = np.mean([self.predictions[:, :, z] for z, ank in enumerate(self.anchors[2])
                        if ank <= idx < ank + self.n], axis=0)
        self._animate_layer(self.slices.getlayer(idx) / self.max_val, pred, mode=mode)
        plt.show()


def tif_save(slices, filename):
    """
    Re-save img as tif
    :param slices: Sliceloader of an image with shape Height x Width x Depth
    """
    try:
        mode = slices.img_stack.mode
    except AttributeError:
        print("Couldn't infer mode - using \"L\"")
        mode = "L"
    if mode == "F":
        dtype = "uint16"
    elif mode == "L":
        dtype = "uint8"
    else:
        raise ValueError("No idea what dtype this needs to be")

    # Create file on first slices of first block
    with tifffile.TiffWriter(filename) as tif:
        tif.save(slices.getlayer(0).astype(dtype), photometric="minisblack")
    with tifffile.TiffWriter(filename, append=True) as tif:
        for i in range(1, slices.size()):
            try:
                tif.save(slices.getlayer(i).astype(dtype), photometric="minisblack")
            except ValueError:
                print("not fully saved")
                break


def swap_save(slices, filename, axis=0, block_size=100):
    """
    Re-save img as tif
    :param slices: Sliceloader of an image with shape Height x Width x Depth
    :param axis:    which axs to swap with depth, default is Height (only one tested atm)
    """
    try:
        mode = slices.img_stack.mode
    except AttributeError:
        print("Couldn't infer mode - using \"L\"")
        mode = "L"
    if mode == "F":
        dtype = "uint16"
    elif mode == "L":
        dtype = "uint8"
    else:
        raise ValueError("No idea what dtype this needs to be")

    block_anks = block_size * np.arange(slices.size() // block_size + 1)
    if slices.size() % block_size != 0:
        block_anks = np.append(block_anks, block_anks[-1] + slices.size() % block_size)
    num_blocks = len(block_anks) - 1

    # old_depth x Width x new_depth (height)
    slc_shape = slices.getlayer(0).shape
    frame_shape = [slices.size(), slc_shape[1-axis], slc_shape[axis]]
    # old_depth x Width x block_size (height)
    block_shape = frame_shape.copy()
    block_shape[2] = block_size

    # Create file on first slices of first block
    frames = np.zeros(block_shape, dtype)
    for i in range(slices.size()):
        # Swap new_depth to the back
        frames[i, :, :] = np.swapaxes(slices.getlayer(i), axis, 1)[:, block_anks[0]:block_anks[1]]
    print("[%.1f %%]" % (100 / num_blocks))
    # Image.fromarray(frames[:, :, 0]).convert(mode).save(filename, bigtiff=True, format="tiff")
    with tifffile.TiffWriter(filename) as tif:
        tif.save(frames[:, :, 0], photometric="minisblack")
    with tifffile.TiffWriter(filename, append=True) as tif:
        # Rest of first block
        for img in range(1, block_size):
            tif.save(frames[:, :, img], photometric="minisblack")

        # num_blocks passes
        for b in range(1, num_blocks):
            # collect portion
            if b == num_blocks - 1:
                block_shape[2] = slices.size() % block_size
            frames = np.zeros(block_shape, dtype)
            for i in range(slices.size()):
                # Swap new_depth to the back
                frames[i, :, :] = np.swapaxes(slices.getlayer(i), axis, 1)[:, block_anks[b]:block_anks[b+1]]
            print("[%.1f %%]" % (100 * (b + 1) / num_blocks))

            for img in range(block_shape[2]):
                tif.save(frames[:, :, img], photometric="minisblack")


if __name__ == "__main__":
    # test, save, bits = ["D:/Data/Beton/Real/%sHPC1-crop-around-crack.tif", "E:/Coding/Python/Betonrisse/results/data/HPC_new/%s.mp4", 16]
    test, save, bits = ["D:/Data/Beton/Real-1/%s180207_UNI-KL_Test_Ursprungsprobe_Unten_PE_25p8um-jpg-xy.tif", "E:/Coding/Python/Betonrisse/results/data/Tube/%s.mp4", 8]
    if test.endswith(".tif"):
        slices = TifLoader(test % "", 100, 0)
    elif test.endswith(".tar"):
        test_tif = test.replace(".tar", ".tif")
        slices = TarLoader(test % "", 100, 0)
        tif_save(slices, test_tif % "")
    # swap_save(slices, test % "rot0_", axis=0)
    # swap_save(slices, test % "rot1_", axis=1)
    # BetonImg(test % "", max_val=2**bits).animate(mode="none", filename=save % "norm")
    BetonImg(test % "rot0_", max_val=2**bits).animate(mode="none", filename=save % "rot0")
    BetonImg(test % "rot1_", max_val=2**bits).animate(mode="none", filename=save % "rot1")

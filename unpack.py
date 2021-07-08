import io, os
import tarfile
from PIL import Image

import numpy as np


def unpack(img_dir, dest):
    """
    unpack tar and convert to .npy files
    """
    # Read sort and filter file names
    # Only use .jpg
    tf = tarfile.open(img_dir)
    img_names = sorted(filter(lambda k: k.endswith("jpg"), tf.getnames()))

    for name in img_names:
        # read
        image = tf.extractfile(name)
        image = image.read()
        # convert to greyscale array
        image = Image.open(io.BytesIO(image)).convert("L")
        image = np.array(image)
        # save
        np.save(dest + name.split("/")[-1][:-4] + ".npy", image)


def split_overlap(arr, d, axis, overlap=True):
    """
    split an array into chunks of size dxdx...xd with 50 % overlap, discarding excess
    """
    new_size = [d * (x // d) for x in arr.shape]
    idx = tuple([slice(0, e) for e in new_size])
    arr_list = [arr[idx]]

    if overlap:
        new_size = [d // 2 + d * ((x - d // 2) // d) if x >= 2 * d else x for x in arr.shape]
        idx_2 = tuple([slice(d // 2, e) if e > d else slice(0, e) for e in new_size])
        arr_list += [arr[idx_2]]

    for axs in axis:
        arr_list = [a for l in map(lambda a: np.split(a, a.shape[axs] / d, axis=axs), arr_list) for a in l]

    return arr_list


def cut(img_dir, dest, d=100):
    """
    read directory of xy slices and make overlapping xyz cubes of size dxdxd with 50 % overlap
    """
    img_names = sorted(filter(lambda k: k.endswith("npy"), os.listdir(img_dir)))

    i = 0
    k = d // 2
    first_half = [np.load(img_dir + img_names[i]) for i in range(k)]
    for z in range(len(img_names) // k - 1):
        second_half = [np.load(img_dir + img_names[i]) for i in range(k * (z+1), k * (z+2))]
        full = np.stack(first_half + second_half, axis=2)
        for arr in split_overlap(full, d, axis=(0, 1)):
                np.save(dest + "%d.npy" % i, arr)
                i += 1
        first_half = second_half


if __name__ == "__main__":
    unpack("D:Data/Beton/Real-1/180207_UNI-KL_Test_Ursprungsprobe_Unten_PE_25p8um-jpg-xy.tar", "D:Data/Beton/Real-1/xy-npy/")
    cut("D:Data/Beton/Real-1/xy-npy/", "D:Data/Beton/Real-1/xyz-100-npy/", d=100)

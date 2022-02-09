from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from data.data_tools import train_test_dataloader
from data.data_transforms import ToTensor, Normalize, Normalize_each, Random_rotate_flip_3d
from data.semisynthetic_data import SemiSynthdata
from data.real_data import Betondata
from paths import *


"""
Create suitable datasets for training and testing
"""


def Betondataset(type, binary_labels=True, test=0.2, **kwargs):
    """
    create dataset from hard-coded presets

    :param type: with labels:
                 "synth", "semisynth", "semisynth-inf", "nc-val", "semisynth-inf-val"
                 without labels:
                 "nc", "hpc", "hpc-riss"
    :param test: percentage of data to hold out in test set. If =0 no test set is returned
    :param kwargs: kwargs for dataloader, e.g. norm, batch_size, shuffle, num_workers, ...
    """

    # stored synthetic
    if type == "synth":
        norm = kwargs.pop("norm", (0, 1))
        data = Betondata(img_dirs=SYNTH_PATH + "input/", label_dirs=SYNTH_PATH + "label/",
                         binary_labels=binary_labels,
                         transform=transforms.Compose([
                             transforms.Lambda(ToTensor()),
                             transforms.Lambda(Random_rotate_flip_3d(cache=not binary_labels))
                         ]),
                         data_transform=transforms.Lambda(Normalize(*norm)))
    # stored semi-synthetic
    elif type == "semisynth":
        norm = kwargs.pop("norm", (0, 1))
        data = Betondata(img_dirs=SEMISYNTH_PATHS_INPUT,
                         label_dirs=SEMISYNTH_PATHS_LABEL,
                         binary_labels=binary_labels,
                         transform=transforms.Compose([
                             transforms.Lambda(ToTensor()),
                             transforms.Lambda(Random_rotate_flip_3d(cache=not binary_labels))
                         ]),
                         data_transform=transforms.Lambda(Normalize(*norm)))
        kwargs.pop("shuffle")
        # fixed test = 0.2
        test = [x for a, b in [(0, 160), (200, 280), (300, 460), (500, 580), (600, 760), (800, 880)]
                for x in list(range(a, b))]
        train = [x for x in range(900) if x not in test]
        return [DataLoader(data, sampler=SubsetRandomSampler(idxs), **kwargs) for idxs in [train, test]]
    # semi-synthetic, just in time
    elif type == "semisynth-inf":
        confidence = kwargs.pop("confidence", 1)
        norm = kwargs.pop("norm", (0, 1))
        data = SemiSynthdata(n=100, size=1000, width=[1, 3, 5], num_cracks=[0, 1, 2],
                             binary_labels=binary_labels,
                             confidence=confidence,
                             transform=transforms.Compose([
                                 transforms.Lambda(Random_rotate_flip_3d()),
                                 transforms.Lambda(Normalize(*norm))
                                 # transforms.Lambda(normalize_each())
                             ]))
    # High Performance Concrete
    elif type == "hpc":
        # max: 206
        # mean: 32.69
        # std: 4.98
        norm = kwargs.pop("norm", (32.69, 4.98))
        data = Betondata(img_dirs="D:Data/Beton/HPC/xyz-100-npy/", binary_labels=binary_labels,
                         transform=transforms.Compose([
                            transforms.Lambda(ToTensor()),
                            transforms.Lambda(Normalize(*norm)),
                            transforms.Lambda(Random_rotate_flip_3d())
                         ]))
    # Normal Concrete
    elif type == "nc":
        # max: 243
        # mean: 25.28
        # std: 3.54
        norm = kwargs.pop("norm", (25.28, 3.54))
        data = Betondata(img_dirs="D:Data/Beton/HPC/xyz-100-npy/", binary_labels=binary_labels,
                         transform=transforms.Compose([
                            transforms.Lambda(ToTensor()),
                            transforms.Lambda(Normalize(*norm)),
                            transforms.Lambda(Random_rotate_flip_3d())
                         ]))
    # High Performance Concrete (only cracks)
    elif type == "hpc-riss" or "hpc-crack":
        # max:
        # mean: 33.24
        # std: 6.69
        norm = kwargs.pop("norm", (33.24, 6.69))
        data = Betondata(img_dirs="D:Data/Beton/HPC/riss/", binary_labels=binary_labels,
                         transform=transforms.Compose([
                            transforms.Lambda(ToTensor()),
                            transforms.Lambda(Normalize(*norm)),
                            transforms.Lambda(Random_rotate_flip_3d())
                         ]))
    # Normal Concrete (with labels)
    elif type == "nc-val":
        # Created using:
        # [np.save("D:/Data/Beton/NC/test/label/%d.npy" % i, np.array([[[x]]]))
        # for i, x in zip([101, 55, 56, 58, 60, 65, 85, 95, 97, 99], [0,1,1,1,0,0,0,1,1,0])]
        norm = kwargs.pop("norm", (0, 255))
        data = Betondata(img_dirs=NC_TEST_PATH + "input/",
                         label_dirs=NC_TEST_PATH + "/label/",
                         binary_labels=binary_labels,
                         transform=transforms.Compose([
                             transforms.Lambda(ToTensor()),
                             transforms.Lambda(Normalize(*norm))
                             # transforms.Lambda(normalize_each())
                         ]))
    # train + validation to be used
    elif type == "semisynth-inf-val":
        return [Betondataset("semisynth-inf", test=0, **kwargs), Betondataset("nc-val", test=0, **kwargs)]
    else:
        raise ValueError("Dataset not supported")

    if test > 0:
        return train_test_dataloader(data, test_split=test, **kwargs)
    else:
        return DataLoader(data, **kwargs)

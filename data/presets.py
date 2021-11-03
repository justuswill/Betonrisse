from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from .data_tools import ToTensor, normalize, normalize_each, random_rotate_flip_3d, train_test_dataloader
from .semisynthetic_data import SemiSynthdata
from .data import Betondata


def Betondataset(type, binary_labels=True, test=0.2, **kwargs):
    """
    create dataset from hard-coded presets

    :param type: "synth", "semisynth", "hpc", "hpc-riss" supported
    :param test: percentage of data to hold out in test set. If =0 no test set is created
    :param kwargs: args for dataloader, e.g. batch_size, shuffle=False, num_workers, ...
    """

    if type == "synth":
        data = Betondata(img_dirs="D:/Data/Beton/Synth/input/", label_dirs="D:/Data/Beton/Synth/label/",
                         binary_labels=binary_labels,
                         transform=transforms.Compose([
                             transforms.Lambda(ToTensor()),
                             transforms.Lambda(random_rotate_flip_3d())
                         ]),
                         data_transform=transforms.Lambda(normalize(0.5, 1)))
    elif type == "semisynth":
        data = Betondata(img_dirs=["D:/Data/Beton/Semi-Synth/w%d-npy-100/input%s/" %
                                   (w, s) for w in [1, 3, 5] for s in ["", "2"]],
                         label_dirs=["D:/Data/Beton/Semi-Synth/w%d-npy-100/label%s/" %
                                     (w, s) for w in [1, 3, 5] for s in ["", "2"]],
                         binary_labels=binary_labels,
                         transform=transforms.Compose([
                             transforms.Lambda(ToTensor()),
                             transforms.Lambda(random_rotate_flip_3d())
                         ]),
                         data_transform=transforms.Lambda(normalize(0.11, 1)))
        kwargs.pop("shuffle")
        # fixed test = 0.2
        test = [x for a, b in [(0, 160), (200, 280), (300, 460), (500, 580), (600, 760), (800, 880)]
                for x in list(range(a, b))]
        train = [x for x in range(900) if x not in test]
        return [DataLoader(data, sampler=SubsetRandomSampler(idxs), **kwargs) for idxs in [train, test]]
    elif type == "semisynth-inf":
        data = SemiSynthdata(n=100, size=1000, width=[1, 3, 5], num_cracks=[0, 1, 2],
                             binary_labels=binary_labels,
                             transform=transforms.Compose([
                                 transforms.Lambda(random_rotate_flip_3d()),
                                 transforms.Lambda(normalize_each())
                             ]))
    elif type == "hpc":
        # max: 206
        # mean: 32.69
        # std: 4.98
        data = Betondata(img_dirs="D:Data/Beton/HPC/xyz-100-npy/", binary_labels=binary_labels,
                         transform=transforms.Compose([
                            transforms.Lambda(ToTensor()),
                            transforms.Lambda(normalize(32.69, 4.98)),
                            transforms.Lambda(random_rotate_flip_3d())
                         ]))
    elif type == "nc":
        # max: 243
        # mean: 25.28
        # std: 3.54
        norm = kwargs.pop("norm", (25.28, 3.54))
        data = Betondata(img_dirs="D:Data/Beton/HPC/xyz-100-npy/", binary_labels=binary_labels,
                         transform=transforms.Compose([
                            transforms.Lambda(ToTensor()),
                            transforms.Lambda(normalize(*norm)),
                            transforms.Lambda(random_rotate_flip_3d())
                         ]))
    elif type == "hpc-riss":
        data = Betondata(img_dirs="D:Data/Beton/HPC/riss/", binary_labels=binary_labels,
                         transform=transforms.Compose([
                            transforms.Lambda(ToTensor()),
                            transforms.Lambda(normalize(33.24, 6.69)),
                            transforms.Lambda(random_rotate_flip_3d())
                         ]))
    elif type == "nc-val":
        # [np.save("D:/Data/Beton/NC/test/label/%d.npy" % i, np.array([[[x]]]))
        # for i, x in zip([101, 55, 56, 58, 60, 65, 85, 95, 97, 99], [0,1,1,1,0,0,0,1,1,0])]
        data = Betondata(img_dirs="D:Data/Beton/NC/test/input/",
                         label_dirs="D:Data/Beton/NC/test/label/",
                         binary_labels=binary_labels,
                         transform=transforms.Compose([
                             transforms.Lambda(ToTensor()),
                             # transforms.Lambda(normalize(0, 255))
                             transforms.Lambda(normalize_each())
                         ]))
    elif type == "semisynth-inf-val":
        return [Betondataset("semisynth-inf", test=0, **kwargs), Betondataset("nc-val", test=0, **kwargs)]
    else:
        raise ValueError("Dataset not supported")

    if test > 0:
        return train_test_dataloader(data, test_split=test, **kwargs)
    else:
        return DataLoader(data, **kwargs)
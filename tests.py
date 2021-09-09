import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data import create_synthetic, create_semisynthetic, convert_3d, mean_std
from data import plot_batch, ToTensor, normalize, random_rotate_flip_3d, resize
from data import Betondata, Synthdata, SemiSynthdata


def test_synthetic_data():
    # create_synthetic("D:/Data/Beton/Synth/input/", "D:/Data/Beton/Synth/label/", size=1000)

    data = Synthdata(n=100, size=32, empty=True, noise=True, octaves=1,
                     transform=transforms.Compose([
                         transforms.Lambda(ToTensor()),
                         transforms.Lambda(random_rotate_flip_3d())
                     ]),
                     data_transform=transforms.Lambda(normalize(0.5, 1)))
    dataloader = DataLoader(data, batch_size=8, shuffle=True, num_workers=0)
    # trainloader, testloader = train_test_dataloader(data, batch_size=8, shuffle=True, num_workers=0)

    batch = next(iter(dataloader))
    plot_batch(batch["X"], 2)
    plot_batch(batch["y"], 2)
    plot_batch(batch["X"], 1)
    plot_batch(batch["y"], 1)
    plot_batch(batch["X"], 0)
    plot_batch(batch["y"], 0)
    plt.show()


def test_semisynthetic_data_old():
    # for w in [1, 3, 5]:
    #     for s in ["input", "label"]:
    #         convert_3d("D:/Data/Beton/Semi-Synth/width%d/%s/" % (w, s),
    #                    "D:/Data/Beton/Semi-Synth/w%d-npy-256/%s/" % (w, s))
    # convert_3d("D:/Data/Beton/Semi-Synth/background/", "D:/Data/Beton/Semi-Synth/bg-npy-256/")

    data = Betondata(img_dirs=["D:/Data/Beton/Semi-Synth/w%d-npy-256/input/" % w for w in [3, 5]]
                     + ["D:/Data/Beton/Semi-Synth/bg-npy-256/"],
                     label_dirs=["D:/Data/Beton/Semi-Synth/w%d-npy-256/label/" % w for w in [3, 5]],
                     transform=transforms.Compose([
                         transforms.Lambda(ToTensor()),
                         transforms.Lambda(resize(100)),
                         transforms.Lambda(random_rotate_flip_3d())
                     ]),
                     data_transform=transforms.Lambda(normalize(30, 6.5)))
    dataloader = DataLoader(data, batch_size=8, shuffle=True, num_workers=0)

    batch = next(iter(dataloader))
    plot_batch(batch["X"], 2)
    plot_batch(batch["y"], 2)
    plot_batch(batch["X"], 1)
    plot_batch(batch["y"], 1)
    plot_batch(batch["X"], 0)
    plot_batch(batch["y"], 0)
    plt.show()


def test_semisynthetic_data():
    # for w in [1, 3, 5]:
    #     create_semisynthetic("D:/Data/Beton/Semi-Synth/w%d-npy-100/input/" % w,
    #                          "D:/Data/Beton/Semi-Synth/w%d-npy-100/label/" % w, size=200, width=w, num_cracks=1)
    # for w in [1, 3, 5]:
    #     create_semisynthetic("D:/Data/Beton/Semi-Synth/w%d-npy-100/input2/" % w,
    #                          "D:/Data/Beton/Semi-Synth/w%d-npy-100/label2/" % w, size=100, width=w, num_cracks=2)

    data = SemiSynthdata(n=100, size=1000, width=[1, 3, 5], num_cracks=[0, 0, 1, 2],
                         transform=transforms.Lambda(random_rotate_flip_3d())
                         # data_transform=transforms.Lambda(normalize(0.5, 1))
                         )
    dataloader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    plot_batch(batch["X"], 2)
    plot_batch(batch["y"], 2)
    plot_batch(batch["X"], 1)
    plot_batch(batch["y"], 1)
    plot_batch(batch["X"], 0)
    plot_batch(batch["y"], 0)
    plt.show()


if __name__ == "__main__":
    # test_synthetic_data()
    test_semisynthetic_data()

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from data import BetonImg, Normalize
from paths import *
# from cnn_3d import Net
from models import LegNet1

"""
Prediction and animation of big 3D images
"""

if __name__ == "__main__":
    import tracemalloc
    tracemalloc.start()
    torch.cuda.empty_cache()

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    # [0]0-110[255]
    test, bits = TUBE_PATH, 8
    # test = HPC_8_PATH
    # test = NC_8_PATH
    # [0]1902-27301[65536]
    # test = HPC_16_PATH
    # test = HPC_8_PATH

    load = "results/pred_tube_same.npy"

    # Net works best on data in [0 255]?
    data = BetonImg(test, load=load, max_val=2**bits,
                    transform=transforms.Lambda(Normalize(0.11, 1)))

    # Net
    # net = Net(layers=1, dropout=0.0).to(device)
    net = LegNet1(layers=1).to(device)
    net.load_state_dict(torch.load("checkpoints/shift_0_11/netcnn_l1p_epoch_5.cp", map_location=device))

    data.predict(net, device)
    layer = 1100
    data.plot_layer(layer, mode="cmap")
    data.plot_layer(layer, mode="cmap-alpha")
    data.plot_layer(layer, mode="clas")
    plt.show()
    # data.animate(mode="cmap")

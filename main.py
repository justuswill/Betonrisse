import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from data import BetonImg, Normalize, Normalize_median, Normalize_min_max
from paths import *
from models import LegNet1, Net, Net_from_Seg

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
    # test, bits = HPC_8_PATH, 8
    # test, bits = NC_8_PATH, 8
    # [0]1902-27301[65536]
    # test = HPC_16_PATH
    # test = HPC_8_PATH

    load = None  # "results/pred_tube_adam.npy"

    # Net works best on data in [0 255]?
    data = BetonImg(test, overlap=50, load=load, max_val=2**bits - 1,
                    transform=transforms.Compose([
                        transforms.Lambda(Normalize_min_max()),
                        transforms.Lambda(Normalize(0, 0.4))
                    ]))

    # Net
    net = Net(layers=1, dropout=0.1, kernel_size=5).to(device)
    # net = LegNet1(layers=1).to(device)
    # net = Net_from_Seg(layers=3, dropout=0.1)
    net.load_state_dict(torch.load("checkpoints/current_epoch_3", map_location=device))

    skip = 30  # 19
    data.predict(net, device, head=1, skip=skip)
    layer = 1525
    data.plot_layer(layer, mode="cmap")
    data.plot_layer(layer, mode="cmap-alpha")
    data.plot_layer(layer, mode="clas")
    plt.show()
    # data.animate(mode="cmap")

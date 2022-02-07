import torch
import torchvision.transforms as transforms

from data import BetonImg, Normalize
from paths import *
# from cnn_3d import Net
from models.legacy import LegNet1

"""
Prediction and animation of big 3D images
"""

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    # [0]0-110[255]
    # test = TUBE_PATH
    # test = HPC_8_PATH
    # test = NC_8_PATH
    # [0]1902-27301[65536]
    test = HPC_16_PATH
    # test = HPC_8_PATH

    load = "results/pred.npy"

    # Net works best on data in [0 255]?
    tgt_max_val = 2**16
    scale_from_tgt = 1
    data = BetonImg(test, load=load, max_val=tgt_max_val * scale_from_tgt,
                    transform=transforms.Lambda(Normalize(0, scale_from_tgt)))

    # Net
    # net = Net(layers=1, dropout=0.0).to(device)
    net = LegNet1(layers=1).to(device)
    net.load_state_dict(torch.load("checkpoints/shift_0_11/netcnn_l1p_epoch_5.cp", map_location=device))

    # data.predict(net, device)
    # data.plot_layer(300, mode="clas")
    # data.plot_layer(400, mode="cmap")
    # data.plot_layer(300, mode="cmap-alpha")
    # data.animate(mode="cmap")

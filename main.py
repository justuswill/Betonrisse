import torch
import torchvision.transforms as transforms

from data import BetonImg, normalize
# from cnn_3d import Net
from models.legacy import LegNet1

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    # test = "D:/Data/Beton/Real-1/180207_UNI-KL_Test_Ursprungsprobe_Unten_PE_25p8um-jpg-xy.tar"
    # [0]0-110[255]: 23/36 -> 1
    # test = "D:/Data/Beton/Semi-Synth/210224_UNI_KL_Kiesche_selbstheilend_gerissen_HPC8_22p7um_10windowWidth.tif"
    # [0]1902-27301[65536]: 6610/6654 -> 250
    test = "D:/Data/Beton/Real/rot0_HPC1-crop-around-crack.tif"
    # test = "D:/Data/Beton/Real/NC2-crop-around-crack.tif"
    # test = "D:/Data/Beton/Real/210225_UNI_KL_Kiesche_selbstheilend_gerissen_HPC3_22p7um_10windowWidth.iass.tif"

    load = "results/pred.npy"

    # Net works best on data in [0 255]
    tgt_max_val = 2**16
    scale_from_tgt = 1
    data = BetonImg(test, load=load, max_val=tgt_max_val * scale_from_tgt,
                    transform=transforms.Lambda(normalize(0, scale_from_tgt)))

    # Net
    # net = Net(layers=1, dropout=0.0).to(device)
    net = LegNet1(layers=1).to(device)
    net.load_state_dict(torch.load("checkpoints/shift_0_11/netcnn_l1p_epoch_5.cp", map_location=device))

    # data.predict(net, device)
    # data.plot_layer(300, mode="clas")
    data.plot_layer(400, mode="cmap")
    # data.plot_layer(300, mode="cmap-alpha")
    # data.animate(mode="cmap")

"""
Paths to relevant data etc.
"""

# animation
FFMPEG_PATH = 'D:/ffmpeg/bin/ffmpeg.exe'

# data

# with labels, everything .npy
# saved synthetic, has folder input/ and label/
SYNTH_PATH = "D:/Data/Beton/Synth/"
# saved semisynthetic, multiple folders, different widths and number of cracks
SEMISYNTH_PATHS_INPUT = ["D:/Data/Beton/Semi-Synth/w%d-npy-100/input%s/" % (w, s) for w in [1, 3, 5] for s in ["", "2"]]
SEMISYNTH_PATHS_LABEL = ["D:/Data/Beton/Semi-Synth/w%d-npy-100/label%s/" % (w, s) for w in [1, 3, 5] for s in ["", "2"]],
# saved real (NC) [Justus], has folder input/ and label/
NC_TEST_PATH = "D:Data/Beton/NC/test/"
# saved real (NC / HPC) [Shai], has folder input/ and label/
REAL_TEST_PATH = "D:Data/Beton/Test-Real/"
# backgrounds for semisynthetic
BG_PATH = "D:/Data/Beton/Semi-Synth/bg-npy-256/"

# Big Images, no labels
TUBE_PATH = "D:/Data/Beton/Real-1/180207_UNI-KL_Test_Ursprungsprobe_Unten_PE_25p8um-jpg-xy.tar"
HPC_8_PATH = "D:/Data/Beton/Real/210225_UNI_KL_Kiesche_selbstheilend_gerissen_HPC3_22p7um_10windowWidth.iass.tif"
NC_8_PATH = "D:/Data/Beton/Real/210226_UNI_KL_Kiesche_selbstheilend_gerissen_NC12_22p7um_10windowWidth.iass.tif"
HPC_16_PATH = "D:/Data/Beton/Real/rot0_HPC1-crop-around-crack.tif"
NC_16_PATH = "D:/Data/Beton/Real/NC2-crop-around-crack.tif"

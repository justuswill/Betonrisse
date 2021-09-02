import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import gaussian_filter

"""
Brownian surface generation adapted from
https://gist.github.com/radarsat1/6f8b9b50d1ecd2546d8a765e8a144631
"""


# embedding of covariance function on a [0,R]^2 grid
def rho(x, y, R, alpha):

    if alpha <= 1.5:
        # alpha=2*H, where H is the Hurst parameter
        beta = 0
        c2 = alpha/2
        c0 = 1-alpha/2
    else:
        # parameters ensure piecewise function twice differentiable
        beta = alpha*(2-alpha)/(3*R*(R**2-1))
        c2 = (alpha-beta*(R-1)**2*(R+2))/2
        c0 = beta*(R-1)**3+1-c2

    # create continuous isotropic function
    r = np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    if r <= 1:
        out = c0-r**alpha+c2*r**2
    elif r <= R:
        out = beta*(R-r)**3/r
    else:
        out = 0

    return out, c0, c2


# The main control is the Hurst parameter: H should be between 0 and
# 1, where 0 is very noisy, and 1 is smoother.
def brownian_surface(N=1000, H=0.95):
    R = 2  # [0,R]^2 grid, may have to extract only [0,R/2]^2

    # size of grid is m*n; covariance matrix is m^2*n^2
    M = N

    # create grid for field
    tx = np.linspace(0, R, M)
    ty = np.linspace(0, R, N)
    rows = np.zeros((M,N))

    for i in range(N):
        for j in range(M):
            # rows of blocks of cov matrix
            rows[j,i] = rho([tx[i], ty[j]],
                            [tx[0], ty[0]],
                            R, 2*H)[0]

    BlkCirc_row = np.vstack(
        [np.hstack([rows, rows[:, -1:1:-1]]),
         np.hstack([rows[-1:1:-1, :], rows[-1:1:-1, -1:1:-1]])])

    # compute eigen-values
    lam = np.real(np.fft.fft2(BlkCirc_row))/(4*(M-1)*(N-1))
    lam = np.sqrt(lam)

    # generate field with covariance given by block circular matrix
    Z = np.vectorize(complex)(np.random.randn(2*(M-1), 2*(M-1)),
                              np.random.randn(2*(M-1), 2*(M-1)))
    F = np.fft.fft2(lam*Z)
    F = F[:M, :N]  # extract sub-block with desired covariance

    out, c0, c2 = rho([0, 0], [0, 0], R, 2*H)

    field = np.real(F)
    field = field - field[0, 0]  # set field zero at origin

    # make correction for embedding with a term c2*r^2
    field = field + np.kron(np.array([ty]).T * np.random.randn(), np.array([tx]) * np.random.randn())*np.sqrt(2*c2)

    field = field[:N//2, :M//2]
    return field


def generate_crack(n=100, width=3, H=0.99, sigma=0.5):
    """
    Generate a random crack surface in 3d using a brownian surface

    :param width: width of cracks
    :param H: Hurst index; the large H, the smoother the surface
    :param n: size of output
    :param sigma: label smoothing with a gaussian filter, set to 0 for no smoothing
    """
    gt = np.zeros([n, n, n])
    field = brownian_surface(2*n, H)

    # discretization
    min_val = np.nanmin(field)  # min and max values for scaling
    max_val = np.nanmax(field)
    m = max(np.abs(min_val), np.abs(max_val))
    if m > 1:
        c = (n / 2 - 1) / m
    else:
        c = n / 2
    field = np.floor(c * field)

    # scale and round field array starting from the plane in the center at z position n/2, the values in field
    # determine the direction (depending on the sign) and the number of steps to the entries that get label 1
    gt[:, :, int(n / 2)] = field
    for i in range(n):
        for j in range(n):
            z = gt[i, j, int(n / 2)]
            if z == 0:
                gt[i, j, int(n / 2)] = 1
            else:
                gt[i, j, int(n / 2)] = 0
                gt[i, j, int(n / 2 + z)] = 1

    # Make crack wider
    struct = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                       [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                       [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    for _ in range(int(np.round(width - 1 / 2))):
        gt = binary_dilation(gt, struct)

    # smooth labels around edges
    gt = gaussian_filter(gt.astype(np.float32), sigma=sigma)

    return gt


if __name__ == "__main__":
    img = crack(width=3, n=100)
    for i in range(0, 100, 5):
        plt.imshow(img[i, :, :], cmap='Greys')
        plt.pause(0.1)
    plt.show()

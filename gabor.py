#####
#
#  Make feature extraction repeatable process between
#  model creation / training and prediction
#
#####

import cv2 as cv2
import numpy as np
import pandas as pd


def extract_features(img: np.ndarray) -> pd.DataFrame:
    #  do stuff
    df = pd.DataFrame()
    df['red'] = img[:, :, 2].reshape(-1)
    df['green'] = img[:, :, 1].reshape(-1)
    df['blue'] = img[:, :, 0].reshape(-1)

    return df
    pass

def get_gabor_bank(kernel_size=5):
    """Returns a bank of gabor kernels, determined by kernel_size"""
    l = []
    for theta in range(2):
        theta = theta / 4 * np.pi
        for sigma in (3, 5):
            for lam in np.arange(0, np.pi, np.pi / 4.):
                for gamma in (0.05, 0.5):
                    kernel = cv2.getGaborKernel(
                        (kernel_size, kernel_size),
                        sigma,
                        theta,
                        lam,
                        gamma,
                        0,
                        ktype=cv2.CV_32F)
                    l.append(kernel)
    return np.asarray(l)


if __name__ == "__main__":
    kernels = get_gabor_bank(3)
    print(np.shape(kernels))

    for kernel in kernels:
        print(np.shape(kernel))
        print(kernel)

    img = cv2.imread('images/7.jpg')
    df = extract_features(img)

    print(df.head())


#####
#
#  Make feature extraction repeatable process between
#  model creation / training and prediction
#
#####

import cv2 as cv2
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.filters import sobel
from skimage.filters import scharr

def extract_features(img: np.ndarray) -> pd.DataFrame:
    """ input an image, returns a dataframe of 1d arrays corresponding
        to each filter """
    df = pd.DataFrame()
    df['red'] = img[:, :, 2].reshape(-1)
    df['green'] = img[:, :, 1].reshape(-1)
    df['blue'] = img[:, :, 0].reshape(-1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    df['gray'] = img_gray.reshape(-1)
    kerns = get_gabor_bank(3)
    num = 1
    for kern in kerns:
        img_temp = cv2.filter2D(img_gray, cv2.CV_8UC3, kern)
        df['Gabor' + str(num)] = img_temp.reshape(-1)
        num = num + 1
    img_gaussian = nd.gaussian_filter(img_gray, sigma=3).reshape(-1)
    df['gaussian'] = img_gaussian.reshape(-1)
    img_entropy = entropy(img_gray, disk(1))
    df['entropy'] = img_entropy.reshape(-1)
    img_sobel = sobel(img_gray)
    df['sobel'] = img_sobel.reshape(-1)
    img_scharr = scharr(img_gray)
    df['scharr'] = img_scharr.reshape(-1)
    return df


def get_gabor_bank(kernel_size: int = 3) -> np.ndarray:
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
    print(df.tail())


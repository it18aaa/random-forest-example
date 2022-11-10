import pickle
import cv2 as cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.filters import sobel
from skimage.filters import scharr
from scipy import ndimage as nd

filename = 'RandomForestClassifier.pkl'
model = pickle.load(open(filename, 'rb'))

# TODO: turn into a class
#
def extract_features(img):
    img_red = img[:, :, 2]
    img_green = img[:, :, 1]
    img_blue = img[:, :, 0]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_entropy = entropy(img_gray, disk(1))
    img_gaussian = nd.gaussian_filter(img_gray, sigma=3)
    img_entropy2 = entropy(img_gaussian, disk(2))
    img_sobel = sobel(img_gray)
    img_scharr = scharr(img_gray)

    ksize = 3
    sigma = 3
    theta = 1 * np.pi / 4
    lam = 1 * np.pi / 4
    gamma = 0.5
    phi = 0
    kernel1 = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, phi, ktype=cv2.CV_32F)
    kernel2 = cv2.getGaborKernel((3, 3), 3, 1 * np.pi / 4, .5 * np.pi / 4, 0.1, .5, ktype=cv2.CV_32F)

    img_gabor_1 = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel1)
    img_gabor_2 = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel2)
    df = pd.DataFrame()
    df['Red'] = img_red.reshape(-1)
    df['Green'] = img_green.reshape(-1)
    df['Blue'] = img_blue.reshape(-1)
    df['Gray'] = img_gray.reshape(-1)
    df['Entropy'] = img_entropy.reshape(-1)
    df['Scharr'] = img_scharr.reshape(-1)
    df['Sobel'] = img_sobel.reshape(-1)
    df['Gabor1'] = img_gabor_1.reshape(-1)
    df['Gabor2'] = img_gabor_2.reshape(-1)
    return df


scale_percent = 20

images = 2

f, axar = plt.subplots(images, 2)

for i in range(1, images+1):
    img = cv2.imread('images/' + str(i) + '.jpg')
    dim = int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)
    image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    features = extract_features(image)
    prediction = model.predict(features)
    shape = image[:, :, 0]
    img_segmented = prediction.reshape(shape.shape)
    axar[i - 1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axar[i - 1, 1].imshow(img_segmented, cmap='jet')

plt.show()

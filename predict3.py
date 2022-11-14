import pickle
import cv2 as cv2
from matplotlib import pyplot as plt

from gabor import extract_features

filename = 'RandomForestClassifier3.pkl'
model = pickle.load(open(filename, 'rb'))

scale_percent = 40

images = 9

f, axar = plt.subplots(images, 2)

for i in range(1, images+1):
    img = cv2.imread('images/' + str(i) + '.jpg')
    dim = int(img.shape[1] * scale_percent / 100), \
          int(img.shape[0] * scale_percent / 100)
    image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    features = extract_features(image)
    prediction = model.predict(features)
    shape = image[:, :, 0]
    img_segmented = prediction.reshape(shape.shape)
    axar[i - 1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axar[i - 1, 1].imshow(img_segmented, cmap='jet')


    plt.imsave(fname='images/results/' + str(i) + '.png', arr=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imsave(fname='images/results/' + str(i) + '_segmented.png',
               arr=img_segmented,
               cmap='plasma')

plt.show()

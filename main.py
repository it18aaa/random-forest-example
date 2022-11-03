import numpy as np
import cv2 as cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.filters import sobel
from skimage.filters import scharr
from skimage.transform import resize
from scipy import ndimage as nd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

img = cv2.imread('images/7.jpg')
img_label = cv2.imread('images/7l.jpg')

# resize it so we can play with the image
scale_percent = 20
dim = int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
# img_label = cv2.resize(img_label, dim, interpolation=cv2.INTER_AREA)

# resize labels without interpolation, which adds intensities we dont want
img_label = resize(img_label, (dim[1], dim[0]), mode='edge', anti_aliasing=False, anti_aliasing_sigma=None,
                   preserve_range=True, order=0)

# opencv is BGR, separate out individual channels...
img_red = img[:, :, 2]
img_green = img[:, :, 1]
img_blue = img[:, :, 0]
img_label = img_label[:, :, 2]

# entropy filter only works on grayscale?
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_entropy = entropy(img_gray, disk(1))

img_gaussian = nd.gaussian_filter(img_gray, sigma=3)

# stacking filters ->
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

# cv2.imshow('gabor 1', img_gabor_1)
# cv2.imshow('gabor 2', img_gabor_2)
# cv2.imshow('original', img)
# cv2.imshow('img_red', img_red)
# cv2.imshow('img_green', img_green)
# cv2.imshow('img_blue', img_blue)
# cv2.imshow('img_gray', img_gray)
# cv2.imshow('entropy', img_entropy)
# cv2.imshow('gaussian', img_gaussian)
# cv2.imshow('sobel', img_sobel)
# cv2.imshow('scharr', img_scharr)
# cv2.imshow('entropy2', img_entropy2)
# cv2.imshow('Labels', img_label)

# flatten the images, so they're flat arrays of pixels
# these columns would be 'features'
# prepare dataframe for machine learning activity

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
labels = img_label.reshape(-1)

# show a histogram of labels
# plt.hist(labels)
# plt.show()

# this curates 'interpolated values' into their label, so
# we have only 4 labels
labels[labels < 50] = 1
labels[(labels >= 50) & (labels < 151)] = 2
labels[(labels > 150) & (labels < 200)] = 3
labels[labels >= 200] = 4

df['Labels'] = labels
print(df)

print('labels are ' + str(np.unique(labels)))

# split dataset into labels and data
Y = df['Labels']
X = df.drop(labels=['Labels'], axis=1)

print("Value Counts:")
print(Y.value_counts())

# test size is the proportion of the split between training and testing
# random state ensures that the dataset is split
# stratify ensures the resultant datasets include labels in the same proportion as the source
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10, stratify=Y)

model = RandomForestClassifier(n_estimators=20, random_state=30)
model.fit(X_train, Y_train)

prediction_test = model.predict(X_test)
print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))

feature_list = list(X.columns)
feature_importance = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print("Feature Importance:")
print(feature_importance)

# get confusion matrix
matrix = metrics.confusion_matrix(Y_test, prediction_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16, 7))
sns.set(font_scale=1)
sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidth=0.2)
class_names = ['Grass', 'Soil', 'Path','Veg']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# cv2.waitKey()
# cv2.destroyAllWindows()

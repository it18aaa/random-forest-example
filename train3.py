import numpy as np
import cv2 as cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from gabor import extract_features



# work in progress
# load training images and masks - this should be done in array or dict

img = cv2.imread('labels/8.jpg')
img_conifer = cv2.imread('labels/conifer.png')
img_decking = cv2.imread('labels/decking.png')
img_grass1 = cv2.imread('label/grass1.png')
img_grass2 = cv2.imread('label/grass2.png')
img_house1 = cv2.imread('label/house1.png')
img_house2 = cv2.imread('label/house2.png')
img_house3 = cv2.imread('label/house3.png')
img_longgrass1 = cv2.imread('label/longgrass1.png')
img_longgrass2 = cv2.imread('label/longgrass2.png')
img_oak1 = cv2.imread('label/oak1.png')
img_oak2 = cv2.imread('label/oak2.png')
img_path1 = cv2.imread('label/path1.png')
img_path2 = cv2.imread('label/path2.png')
img_river = cv2.imread('label/river.png')
img_road = cv2.imread('label/road.png')
img_roof1 = cv2.imread('label/roof1.png')
img_roof2 = cv2.imread('label/roof2.png')
img_roof3 = cv2.imread('label/roof3.png')
img_trees1 = cv2.imread('label/trees1.png')
img_trees2 = cv2.imread('label/trees2.png')
img_trees3 = cv2.imread('label/trees3.png')
img_vegetation1 = cv2.imread('label/vegetation1.png')
img_vegetation2 = cv2.imread('label/vegetation2.png')
img_vegetation3 = cv2.imread('label/vegetation3.png')




pd.options.display.float_format = '{:,.4f}'.format


# resize it so we can play with the image
scale_percent = 40
dim = int(img.shape[1] * scale_percent / 100), \
      int(img.shape[0] * scale_percent / 100)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# resize labels without interpolation, which adds intensities we dont want
img_label = resize(img_label, (dim[1], dim[0]),
                   mode='edge',
                   anti_aliasing=False,
                   anti_aliasing_sigma=None,
                   preserve_range=True,
                   order=0)
img_label = img_label[:, :, 2]

df = extract_features(img)

labels = img_label.reshape(-1)

# this curates 'interpolated values' into their label, so
# we have only 4 labels,
# TODO:  clean this up, maybe use label-studio and simpler cleaning method.
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


filename = "RandomForestClassifier3.pkl"
pickle.dump(model, open(filename, 'wb'))

# cv2.waitKey()
# cv2.destroyAllWindows()

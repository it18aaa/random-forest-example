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

img = cv2.imread('images/7.jpg')
img_label = cv2.imread('images/7l.jpg')

# resize it so we can play with the image
scale_percent = 20
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


filename = "RandomForestClassifier2.pkl"
pickle.dump(model, open(filename, 'wb'))

# cv2.waitKey()
# cv2.destroyAllWindows()

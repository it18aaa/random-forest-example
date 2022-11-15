import numpy as np
import cv2 as cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
#from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from gabor import extract_features
from consolidate_labels import stack_labels, resize

# work in progress
# load training images and masks - this should be done in array or dict

pd.options.display.float_format = '{:,.4f}'.format
labels_json = "labels/labels.json"
scale_percent = 60
img_label, label_classes = stack_labels(labels_json, scale_percent)

img_label = img_label[:, :, 2]
labels = img_label.reshape(-1)

img = cv2.imread("labels/8.jpg")
img = resize(img, scale_percent)

labels[labels > 12] = 0

df = extract_features(img)
df['Labels'] = labels

# take out the 0 values as these are not labelled!
df = df[df.Labels != 0]

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

model_accuracy = metrics.accuracy_score(Y_test, prediction_test)

feature_list = list(X.columns)
feature_importance = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print("Feature Importance:")
print(feature_importance)

# get confusion matrix
matrix = metrics.confusion_matrix(Y_test, prediction_test)

print(matrix)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16, 7))

sns.set(font_scale=1)
sns.heatmap(matrix, annot=True, annot_kws={'size': 12}, cmap=plt.cm.Greens, linewidth=0.2)

class_names = list(label_classes.values())
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks2, class_names, rotation=45)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix\nAccuracy: ' + str(model_accuracy))
plt.show()

filename = "RandomForestClassifier3.pkl"
pickle.dump(model, open(filename, 'wb'))

# cv2.waitKey()
# cv2.destroyAllWindows()

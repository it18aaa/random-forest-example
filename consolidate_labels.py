import cv2 as cv2
import numpy as np
import json as json
from os import path


# utility method to resize an image
def resize(image: np.ndarray, scale_percent: int) -> np.ndarray:
    dim = int(image.shape[1] * scale_percent / 100), \
          int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


def stack_labels(labels_json: str = "labels/labels.json",
                 scale_percent: int = 100) -> (np.ndarray, dict[str]):
    if path.exists(labels_json):
        with open(labels_json, 'r') as file:
            label_data = json.load(file)
    else:
        raise Exception("Path not found: " + labels_json)

    labelled_image_name = label_data['labelled_image_name']
    label_path = label_data["label_path"]
    label_classes = label_data['classes']
    label_images = label_data['images']
    # get the base image and create an empty label set.
    img = cv2.imread(label_path + labelled_image_name)
    # resize the image if it isnot 100
    if (scale_percent < 100):
        img = resize(img, scale_percent)
    img_labels = np.zeros(shape=img.shape, dtype='uint8')

    for label_image in label_images:
        class_number = label_image['class']
        print("Stacking: " + label_image['name'] + ": " +
              label_classes[str(class_number)] + "(" +
              str(class_number) + ")")
        img_temp = cv2.imread(label_path + label_image['name'])
        if (scale_percent < 100):
            img_temp = resize(img_temp, scale_percent)
        img_temp[img_temp > 0] = class_number
        img_labels = img_labels | img_temp
    return (img_labels, label_classes)


if __name__ == "__main__":
    img_labels, label_classes = stack_labels(scale_percent=20)
    # show the labels in the merged label image
    print(np.shape(img_labels))
    print(np.unique(img_labels))
    for label_class in label_classes:
        print(label_class + " = " + label_classes[label_class])
    counts, bins = np.histogram(img_labels)

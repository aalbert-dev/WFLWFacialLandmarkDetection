import cv2
import os
from tqdm import tqdm
import numpy as np

"""
@author Arjun Albert
@date 4/16/21
@email aalbert@mit.edu
"""

annotations_path = "/home/arjun/Desktop/cv/pa3/WFLW_annotations/"
images_path = "/home/arjun/Desktop/cv/pa3/WFLW_images/"

training_file_name = "list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
testing_file_name = "list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"

name = annotations_path + training_file_name

LOAD_FULL_DATASET = False


def get_img(file_name):
    cv2_img = None
    for i in range(0, 62):
        try:
            img_path = images_path + str(i) + "--" + file_name
            img_path = img_path[:-1]
            maybe_img = cv2.imread(img_path)
            if maybe_img is not None:
                cv2_img = maybe_img
                cv2_img = cv2.resize(cv2_img, (512, 512),
                                     interpolation=cv2.INTER_AREA)
                break
        except:
            pass
    return cv2_img

def get_img_path(file_name):
    for i in range(0, 62):
        img_path = images_path + str(i) + "--" + file_name
        img_path = img_path[:-1]
        if os.path.exists(img_path):
            return img_path
    return None


def get_crop_points(attributes):
    split_attributes = attributes.split(" ")
    split_attributes = split_attributes[196:200]
    split_attributes = np.array(list(map(lambda x: int(x), split_attributes)))
    return split_attributes


def load_dataset():
    with open(name) as f:
        data = []
        row_count = 0
        for row in f:
            if row_count > 200 and not LOAD_FULL_DATASET: break
            row_values = row.split("--")
            img = row_values[1]
            img_path = get_img_path(img)
            if img_path is not None:
                landmarks = get_keypoints(row_values[0])
                crops = get_crop_points(row_values[0])
                data.append((landmarks, img_path, crops))
            row_count += 1
    return data


def get_keypoints(attributes):
    split_attributes = attributes.split(" ")
    split_attributes = split_attributes[:196]
    landmarks = []
    for i in range(0, len(split_attributes), 2):
        landmarks.append([int(float(split_attributes[i])), int(float(split_attributes[i + 1]))])
    return np.array(landmarks)


raw_data = load_dataset()
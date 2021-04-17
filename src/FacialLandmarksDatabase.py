# -*- coding: utf-8 -*-

import sys
import time
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import matplotlib.image as mpimg
from collections import OrderedDict
from skimage import io, transform
from math import *
import xml.etree.ElementTree as ET

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_loader import raw_data
from transforms import Transforms

"""
@author Arjun Albert
@date 4/16/21
@email aalbert@mit.edu
"""

"""
Facial landmark dataset class to hold image file paths,
landmarks, crop locations, and transform class
"""
class FaceLandmarksDataset(Dataset):

    def __init__(self, transform=None):

        self.image_filenames = np.array(list(map(lambda x: x[1], raw_data)))
        self.landmarks = np.array(list(map(lambda x: x[0], raw_data)))
        self.crops = np.array(list(map(lambda x: x[2], raw_data)))
        self.transform = transform

        self.landmarks = np.array(self.landmarks).astype('float32')

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index])
        landmarks = self.landmarks[index]
        crops = self.crops[index]

        if self.transform:
            image, landmarks = self.transform(
                image, landmarks, crops)

        landmarks = landmarks - 0.5
        return image, landmarks

    def search_for_item(self, path):
        count = 0
        for image_filename in self.image_filenames:
            if path in image_filename:
                return count
            count += 1
        return -1

"""
Initialize dataset for importing
"""
dataset = FaceLandmarksDataset(Transforms())
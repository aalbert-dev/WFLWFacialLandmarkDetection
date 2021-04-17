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
from model import Network
from FacialLandmarksDatabase import FaceLandmarksDataset
from transforms import Transforms

"""
@author Arjun Albert
@date 4/16/21
@email aalbert@mit.edu
"""

"""
Initalize dataset with transforms
"""
dataset = FaceLandmarksDataset(Transforms())

"""
Continuously ask user for input images
"""
while True:

    image_index = -1

    """
    Keep asking for images until we find a valid image search criteria
    """
    while image_index < 0:

        image_path = input("Enter the path to the image you would like to use: ")

        image_index = dataset.search_for_item(image_path)

        if image_index < 0: print("Invalid image selection. Please try again.")

    data_sample = dataset[image_index]

    """
    Get the predictions from the saved model on the input image
    """
    with torch.no_grad():

        best_network = Network()
        best_network.cuda()
        best_network.load_state_dict(torch.load(
            '/home/arjun/Desktop/cv/pa3/src/model/model_state.pth'))
        best_network.eval()

        image = data_sample[0]
        landmarks = np.array((data_sample[1] + 0.5) * 224)
        crops = dataset.crops[image_index]

        batch = torch.stack(tuple(image for i in range(0, 64)))
        batch = batch.cuda()

        prediction = (best_network(batch).cpu()[0] + 0.5) * 224

        plt.imshow(transforms.ToPILImage()(image), cmap='gray')
        #plt.scatter(landmarks[:,0], landmarks[:,1], s = 5, c = 'g')
        plt.scatter(prediction[::2], prediction[1::2], s = 5, c = 'r')
        plt.show()
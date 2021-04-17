# -*- coding: utf-8 -*-

import sys
import time
import cv2
import os
import random
import numpy as np
from tqdm import tqdm
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
from FacialLandmarksDatabase import dataset
from model import Network

"""
@author Arjun Albert
@date 4/16/21
@email aalbert@mit.edu
"""


"""
Split dataset into training and validation sets
"""
len_valid_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_valid_set

train_dataset, valid_dataset,  = torch.utils.data.random_split(
    dataset, [len_train_set, len_valid_set])

"""
Shuffle the order of training and validation sets
"""
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=8, shuffle=True, num_workers=4)

"""
Method to print training and validation status
"""
def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " %
                         (step, total_step, loss))
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " %
                         (step, total_step, loss))

    sys.stdout.flush()


"""
Import and send network to GPU
"""
torch.autograd.set_detect_anomaly(True)
network = Network()
network.cuda()

"""
Define loss and optimzation methods
"""
criterion = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)

"""
Training parameters
"""
loss_min = np.inf
num_epochs = 50
train = True
saved_model_path = '/home/arjun/Desktop/cv/pa3/src/model/model_state.pth'

"""
Training process for the network
"""
if train:
    start_time = time.time()
    for epoch in range(1, num_epochs+1):

        """
        Initialize training, validation, and running loss
        """
        loss_train = 0
        loss_valid = 0
        running_loss = 0

        """
        Train network on batches from dataset
        """
        network.train()
        for step in range(1, len(train_loader)+1):

            images, landmarks = next(iter(train_loader))

            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0), -1).cuda()

            predictions = network(images)

            optimizer.zero_grad()

            loss_train_step = criterion(predictions, landmarks)

            loss_train_step.backward()

            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train/step

            print_overwrite(step, len(train_loader), running_loss, 'train')

        """
        Validation step
        """
        network.eval()
        with torch.no_grad():

            for step in range(1, len(valid_loader)+1):

                images, landmarks = next(iter(valid_loader))

                images = images.cuda()
                landmarks = landmarks.view(landmarks.size(0), -1).cuda()

                predictions = network(images)

                loss_valid_step = criterion(predictions, landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(
            epoch, loss_train, loss_valid))
        print('--------------------------------------------------')

        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), saved_model_path)
            print(
                "\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')

    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time()-start_time))


"""
Calculate RSME over the whole dataset
"""
rmse = 0
with torch.no_grad():

    for step in tqdm(range(0, len(dataset))):
        
        """
        Grab a batch of images and landmarks from the dataset
        """
        images, landmarks = next(iter(dataset))

        best_network = Network()
        best_network.cuda()
        best_network.load_state_dict(torch.load(saved_model_path))
        best_network.eval()

        images, landmarks = next(iter(valid_loader))
        
        images = images.cuda()

        predictions = best_network(images).cpu()
        
        """
        Accumulate RMSE for the batch
        """
        for i in range(0, len(landmarks)):

            cur_landmarks = landmarks[i]
            cur_predictions = predictions[i]
            cur_landmarks_tf = torch.flatten(cur_landmarks)

            for prediction, actual in zip(cur_predictions, cur_landmarks_tf):
                rmse += (prediction - actual) ** 2

"""
Report RMSE for the whole dataset
"""
rmse /= len(dataset)
rmse = rmse ** 0.5
print("RMSE: " + str(rmse))
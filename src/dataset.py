from torchvision import datasets, transforms
from transforms import Transforms
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import data_loader
import os
import numpy as np
import cv2


class FaceLandmarksDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        self.raw_data = data_loader.dataset
        self.landmarks = np.array(list(map(lambda x: x[0], self.raw_data)))
        self.images = np.array(list(map(lambda x: x[1], self.raw_data)))
        self.crops = np.array(list(map(lambda x: x[2], self.raw_data)))
        assert(len(self.landmarks) == len(self.images))
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.transform(self.images[index], self.landmarks[index], self.crops[index])


dataset = FaceLandmarksDataset(Transforms())

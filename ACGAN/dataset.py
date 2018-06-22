import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
from PIL import Image

import os
import pandas as pd
from skimage import io, transform
from torchvision import utils

import h5py
import numpy as np

CROP_SIZE = (224, 224)

hdf5_file = 'initials_dataset.hdf5'

def input_transform():
    return transforms.Compose([
        Image.fromarray,
        transforms.Resize([64,64]),
        transforms.ToTensor(),
    ])

def get_dataset(duration=1):
    return DatasetFromFolder(hdf5_file, input_transform=input_transform())

class DatasetFromFolder(data.Dataset):
    def __init__(self, hdf5_file, input_transform=None):
        super().__init__()
        h5 = h5py.File(hdf5_file, 'r')
        self.images = h5['images']
        self.countries = h5['countries']
        self.cities = h5['cities']
        self.names = h5['names']
        self.initials = h5['initials']

        self.input_transform = input_transform

    def __getitem__(self, index):
        im = self.images[index]

        if self.input_transform:
            im = self.input_transform(im)

        return (im, self.initials[index], 
                self.countries[index], self.cities[index], self.names[index])

    def __len__(self):
        return len(self.images)

# class LetterDataset(data.Dataset):
#     def __init__(self, csv_file)

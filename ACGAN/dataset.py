import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
from PIL import Image

import os
import pandas as pd
from skimage import io, transform
from torchvision import utils

from matplotlib import pyplot as plt

import h5py
import numpy as np

CROP_SIZE = (224, 224)

hdf5_file = 'initials_dataset.hdf5'

# For full dataset from the HDF5
def input_transform():
    return transforms.Compose([
        Image.fromarray,
        transforms.Resize([64,64]),
        transforms.ToTensor(),
    ])

# For custom datasets made with extractor.py
def input_transform2():
    return transforms.Compose([
        Image.fromarray,
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])


def get_dataset(duration=1):
    return DatasetFromFolder(hdf5_file, input_transform=input_transform())

def get_letterdataset(csv, root, duration=1):
    return LetterDataset(csv, root, input_transform=input_transform2())


# For the full dataset
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


# For custom dataset containing only one letter
class LetterDataset(data.Dataset):
    def __init__(self, csv_file, root_dir, input_transform=None):
        self.csv = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.input_transform = input_transform

        # deduce the feature we want from the csv file name
        substring = csv_file[:-6]
        underscore = substring.rfind('_')
        self.feature = substring[underscore+1:]

    def __getitem__(self, index):

        # Get the name of the respective image and the path to it
        img_name = self.csv.iloc[index, 0]
        img_path = self.root_dir + img_name

        # get the image and perform the transformations
        im = io.imread(img_path)
        if self.input_transform:
            im = self.input_transform(im)

        # Get the feature to return
        flags = self.csv.iloc[index, 1:].tolist()
        # print(type(flags))
        feat_idx = flags.index(1)
        feature = self.csv.columns.tolist()[feat_idx + 1] # Also need to check if this is in the right datatype

        # Return the images and features and fill the missing features with 0
        if self.feature == 'countries':
            return (im, 0, feature, 0, 0)
        elif self.feature == 'cities':
            return (im, 0, 0, feature, 0)
        else:
            return (im, 0, 0, 0, feature)


    def __len__(self):
        return len(self.csv)

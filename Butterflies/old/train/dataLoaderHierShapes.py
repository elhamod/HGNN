import os
from torch.utils.data import Dataset
import re
import warnings
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from skimage import io
import time
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import pandas as pd
import progressbar
import joblib
from PIL import Image, ImageStat
import copy
import torchvision.transforms.functional as TF
import random
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import random_split
from torch.utils.data.dataset import ConcatDataset
from sklearn.model_selection import train_test_split

shapes={
    "Curved": ["Circle", "Ellipse"],
    "Quadrilateral":["Square", "Rectangle"],
}

def getCatTransform(cat, codec):
    coded_cat = codec.transform([cat])
    return (lambda x: (x + 2*coded_cat, coded_cat))

def to_one_hot(codec, values):
    value_idxs = codec.transform(values)
    return torch.eye(len(codec.classes_))[value_idxs]

testIndexFileName = "testIndex.pkl"
valIndexFileName = "valIndex.pkl"
trainingIndexFileName = "trainingIndex.pkl"

image_subpath = "Shapes2"

from dataset_normalization import dataset_normalization

normalizeFromResnet = True # originally false
        
class ShapeHierDataset(Dataset):
    def __init__(self, params, verbose=False):
        self.imageDimension = 224
        self.n_channels = True
        self.data_root = params["image_path"]
        self.normalization_enabled = True
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

        self.cats_codec = LabelEncoder()
        
        categories = set()
        for cats in shapes:
            categories.add(cats)
        self.cats_codec.fit(list(categories))
        for cats in shapes:
            print(cats, self.cats_codec.transform([cats]))
        
        
        self.sub_datasets = []            
        for cats in shapes:
                
            self.sub_datasets.append(
                    ImageFolder(os.path.join(self.data_root, image_subpath, cats), transform=transforms.Compose(self.getTransforms()), target_transform=getCatTransform(cats, self.cats_codec)) #  loader=None, is_valid_file=None
            )
                    

        self.concat_dataset = ConcatDataset(self.sub_datasets)
    
    def getTransforms(self):
        transformsList = [
            transforms.ToTensor(),
            self.normalizer]
        
        return transformsList

    def __len__(self):
        return len(self.concat_dataset)
    
    def __getitem__(self, idx):       
            
        image, target = self.concat_dataset[idx]
        if torch.cuda.is_available():
            image = image.cuda()
            
#         print("sub_category", target[0]+2*target[1])
#         print('category', target[1])

        return {'image': image, 
                'sub_category': target[0].item(), 
                'category': target[1].item(),} 

def writeFile(folder_name, file_name, obj):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    try:
        with open(file_name, 'wb') as f:
            joblib.dump(obj, f) 
            print('file',file_name,'written')
    except:
        print("Couldn't write pickle", file_name)
        pass
        
def readFile(fullFileName):
    try:
        with open(fullFileName, 'rb') as filehandle:
            loaded = joblib.load(filehandle) 
            print('file',fullFileName,'read')
            return loaded
    except:
        print("Couldn't read pickle", fullFileName)
        pass  
    
class datasetManager:
    def __init__(self, experimentName, verbose=False):
        self.verbose = verbose
        self.data_root = None
        self.experimentName = experimentName
        self.datasetName = None
        self.reset()
    
    def reset(self):
        self.dataset = None
        self.train_loader = None
        self.validation_loader =  None
        self.test_loader = None
    
    def updateParams(self, params):
        datasetName = "sameName"
        if datasetName != self.datasetName:
            self.reset()
            self.params = params
            self.data_root = params["image_path"]
            self.experiment_folder_name = os.path.join(self.data_root, self.experimentName)
            self.dataset_folder_name = os.path.join(self.experiment_folder_name, datasetName)
            self.datasetName = datasetName
        
    def getDataset(self):
        if self.dataset is None:
            print("Creating dataset...")
            self.dataset = ShapeHierDataset(self.params, self.verbose)
            print("Creating dataset... Done.")
        return self.dataset

    # Creates the train/val/test dataloaders out of the dataset 
    def getLoaders(self, readLoadersFromDisk=True):
        if self.dataset is None:
            self.getDataset()

        training_count = self.params["training_count"]        
        validation_count = self.params["validation_count"]
        batchSize = self.params["batchSize"]

        index_fileNames = [trainingIndexFileName, valIndexFileName, testIndexFileName]
        saved_index_file = os.path.join(self.experiment_folder_name, testIndexFileName)
        loader_indices = []
        if not os.path.exists(saved_index_file):
            
            indices_len = len(self.dataset)
            data_loader = torch.utils.data.DataLoader(self.dataset,
                                          batch_size=indices_len,
                                          shuffle=False)
            data_loader_batch = next(iter(data_loader))
            indices = range(indices_len)
            labels = data_loader_batch['sub_category']
            train_indices, test_indices = train_test_split(indices, test_size= 1-training_count-validation_count, 
                                                           stratify=labels.cpu())
#             print(labels)
#             print(test_indices)
            data_loader = torch.utils.data.DataLoader(self.dataset,
                                          batch_size=indices_len,
                                          shuffle=False)
            data_loader_batch = next(iter(data_loader))
            labels_sub = data_loader_batch['sub_category'][train_indices]
#             print(labels_sub)
            train_indices, val_indices = train_test_split(train_indices, test_size= validation_count, 
                                                           stratify=labels_sub.cpu())
#             print(train_indices)
#             print(val_indices)
            
            # save indices
            loader_indices = [train_indices, val_indices, test_indices]
            for i, name in enumerate(index_fileNames):
                fullFileName = os.path.join(self.experiment_folder_name, name)
                writeFile(self.experiment_folder_name, fullFileName, loader_indices[i])

        else:
            # load the pickles
            print("Loading saved indices...")
            for i, name in enumerate(index_fileNames):        
                loader_indices.append(readFile( os.path.join(self.experiment_folder_name, name)))


        # create samplers
#         print(loader_indices[0])
#         print(loader_indices[1])
#         print(loader_indices[2])
        train_sampler = SubsetRandomSampler(loader_indices[0])
        valid_sampler = SubsetRandomSampler(loader_indices[1])
        test_sampler = SubsetRandomSampler(loader_indices[2])

        # create data loaders.
        print("Creating loaders...")
        self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=train_sampler, batch_size=batchSize)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=valid_sampler, batch_size=batchSize)
        self.test_loader = torch.utils.data.DataLoader(copy.copy(self.dataset), sampler=test_sampler, batch_size=batchSize)        
        print("Creating loaders... Done.")

            
        return self.train_loader, self.validation_loader, self.test_loader
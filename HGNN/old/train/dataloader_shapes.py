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
import matplotlib.pyplot as plt
import pandas as pd
import progressbar
import joblib
from PIL import Image, ImageStat
import copy
import random

shuffle_dataset = True

dataset_fileName = "dataset.pkl"
testLoaderFileName = "testLoader.pkl"
valLoaderFileName = "valLoader.pkl"
trainingLoaderFileName = "trainingLoader.pkl"
testIndexFileName = "testIndex.pkl"
valIndexFileName = "valIndex.pkl"
trainingIndexFileName = "trainingIndex.pkl"

subpaths = ["Circles", "Squares"]
num_of_images = 50
data_root = "/data/BGNN_data/Shapes/"
training_count = 33       
validation_count = 7
batchSize = 25

from dataset_normalization import dataset_normalization
        
class ShapesDataset(Dataset):
    def __init__(self):
        self.samples = [] # The list of all samples
        self.imageIndicesPerSpecies = {} # A hash map for fast retreival
        self.imageDimension = 224
        self.n_channels = 3
        self.normalization_enabled = True
        self.augmentation_enabled = False
        self.normalizer = None
        self.transforms = None
        self.composedTransforms = None
        
        

        
        for j, image_subpath in enumerate(subpaths):
            
            for i in range(num_of_images):

                img_full_path = os.path.join(data_root, image_subpath+"/"+str(i)+".jpg")

                sampleInfo = {
                    'fileName': img_full_path,
                    'class': j,
                    'image': io.imread(img_full_path)
                }
                self.samples.append(sampleInfo)

        # Create transfroms
        # Toggle beforehand so we could create the normalization transform. Then toggle back.
        if self.normalization_enabled == True and self.normalizer is None:
            augmentation, normalization = self.toggle_image_loading(augmentation=False, normalization=False)
            self.normalizer = dataset_normalization(self).getTransform()
            self.toggle_image_loading(augmentation, normalization)
    
    def getTransforms(self):
        transformsList = [transforms.ToPILImage(),
              transforms.ToTensor()]   
            
        if self.normalization_enabled:
            transformsList = transformsList + self.normalizer
        
        return transformsList

    def __len__(self):
        return len(self.samples)
    
    def toggle_image_loading(self, augmentation, normalization):
        old = (self.augmentation_enabled, self.normalization_enabled)
        self.augmentation_enabled = augmentation
        self.normalization_enabled = normalization
        self.transforms = None
        return old

    def __getitem__(self, idx):       
        img_species = self.samples[idx]['class']
        image = self.samples[idx]['image']
        
        if self.transforms is None:
            self.transforms = self.getTransforms()
            self.composedTransforms = transforms.Compose(self.transforms)
        image = self.composedTransforms(image)
            
        fileName = self.samples[idx]['fileName']

        if torch.cuda.is_available():
            image = image.cuda()

        return {'image': image, 
                'class': img_species, 
                'fileName': fileName} 
    

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


# Given a model and a dataset, get an example image where trueLabel=trueIndex where predictedIndex=expectedIndex
# The key param decides if we are looking at "species" or "genus"
def getExamples(model, dataset, trueIndex, expectedIndex, num_of_examples=1):
    result = []
    
    # get all examples trueLabel=speciesIndex
    name = dataset.getSpeciesOfIndex(trueIndex)
    examples = dataset.getSpeciesIndices(name)
            
    for example in examples:
        augmentation, normalization = dataset.toggle_image_loading(augmentation=False, normalization=False)
        image = dataset[example]['image'].unsqueeze(0)
        dataset.toggle_image_loading(augmentation, normalization)
        predictionImage = dataset[example]['image'].unsqueeze(0)
        predictedIndex = model(predictionImage)
        predictedIndex = torch.max(predictedIndex.data, 1)[1].cpu().detach().numpy()[0]
        if (predictedIndex == expectedIndex):
            image = image.squeeze()
            predictionImage = predictionImage.squeeze()
            result.append((image, predictionImage))
            if len(result) == num_of_examples:
                break

    return result
    
class datasetManager:
    def __init__(self, experimentName):
        self.suffix = None
        self.experimentName = experimentName
        self.reset()
    
    def reset(self):
        self.dataset = None
        self.train_loader = None
        self.validation_loader =  None
        self.test_loader = None
        
    def getDataset(self):
        saved_dataset_file = os.path.join(data_root, dataset_fileName)
        if self.dataset is None:
            if not os.path.exists(saved_dataset_file):
                self.dataset = ShapesDataset()
                writeFile(data_root, saved_dataset_file, self.dataset)
            else:
                self.dataset = readFile(saved_dataset_file)
        return self.dataset

    # Creates the train/val/test dataloaders out of the dataset 
    def getLoaders(self):
        if self.dataset is None:
            self.getDataset()
            
        if self.train_loader is None:
            loaders = []
            loader_fileNames = [trainingLoaderFileName, valLoaderFileName, testLoaderFileName]

            saved_loader_file = os.path.join(data_root, testLoaderFileName)

            if not os.path.exists(saved_loader_file):

                index_fileNames = [trainingIndexFileName, valIndexFileName, testIndexFileName]
                saved_index_file = os.path.join(data_root, testIndexFileName)
                loader_indices = []
                if not os.path.exists(saved_index_file):
                    train_indices = []
                    val_indices = []
                    test_indices = []

                    # for each species, get indices for different sets
                    for class_ in subpaths:
                        dataset_size = num_of_images

                        # Logic to find solitting indices for train/val/test.
                        # If dataset_size is too small, there will be overlap.
                        if class_ == "Circles":
                            indices = list(range(50))
                        else:
                            indices = list(range(50, 100))
                            
                        # training set should at least be one element
                        split_train = training_count
                        if split_train == 0:
                            split_train = 1
                        # validation set should start from after training set. But if not enough elements, there will be overlap.
                        # At least one element
                        split_validation_begin = split_train if split_train < dataset_size else dataset_size - 1
                        split_validation = (training_count + validation_count) 
                        if split_validation > dataset_size:
                            split_validation = dataset_size
                        if split_validation == split_validation_begin:
                            split_validation_begin = split_validation_begin - 1
                        # test set is the remaining but at least one element.
                        split_test = split_validation if split_validation < dataset_size else dataset_size-1

                        if shuffle_dataset :
                            np.random.seed(int(time.time()))
                            np.random.shuffle(indices)

                        # aggregate indices
                        sub_train_indices, sub_val_indices, sub_test_indices = indices[:split_train], indices[split_validation_begin:split_validation], indices[split_test:]
                        train_indices = train_indices + sub_train_indices
                        test_indices = test_indices + sub_test_indices
                        val_indices = val_indices + sub_val_indices

                    # save indices
                    loader_indices = [train_indices, val_indices, test_indices]
                    for i, name in enumerate(index_fileNames):
                        fullFileName = os.path.join(data_root, name)
                        writeFile(data_root, fullFileName, loader_indices[i])

                else:
                    # load the pickles
                    print("Loading saved indices...")
                    for i, name in enumerate(index_fileNames):        
                        loader_indices.append(readFile( os.path.join(data_root, name)))


                # create samplers
                print(loader_indices[0])
                print(loader_indices[1])
                print(loader_indices[2])
                train_sampler = SubsetRandomSampler(loader_indices[0])
                valid_sampler = SubsetRandomSampler(loader_indices[1])
                test_sampler = SubsetRandomSampler(loader_indices[2])

                # create data loaders.
                self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=train_sampler, batch_size=batchSize)
                self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=valid_sampler, batch_size=batchSize)
                self.test_loader = torch.utils.data.DataLoader(copy.copy(self.dataset), sampler=test_sampler, batch_size=batchSize)
                self.test_loader.dataset.toggle_image_loading(augmentation=False, normalization=self.dataset.normalization_enabled) # Needed so we always get the same prediction accuracy 
                loaders = [self.train_loader, self.validation_loader, self.test_loader]

                # pickle the loaders
                for i, name in enumerate(loader_fileNames):
                    fullFileName = os.path.join(data_root, name)
                    writeFile(data_root, fullFileName, loaders[i])

            else:
                # load the pickles
                print("Loading saved dataloaders...")
                for i, name in enumerate(loader_fileNames):        
                    loaders.append(readFile(os.path.join(data_root,name)))

                self.train_loader = loaders[0]
                self.validation_loader = loaders[1]
                self.test_loader = loaders[2]
            
        return self.train_loader, self.validation_loader, self.test_loader
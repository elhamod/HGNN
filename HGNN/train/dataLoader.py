import os
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import torchvision
import re

from myhelpers.dataset_normalization import dataset_normalization
from myhelpers.color_PCA import Color_PCA

from .configParser import getDatasetName
from .CSV_processor import CSV_processor



num_of_workers = 8

# This object is used as an interface for dataset, normalization, augmentation, and csv_processor.
class FishDataset(Dataset):
    def __init__(self, type_, params, normalizer, color_pca, csv_processor):
        self.imageDimension = params["img_res"] # if None, CSV_processor will load original images
        self.n_channels = 3
        self.data_root, self.suffix  = getParams(params)
        self.augmentation_enabled = False
        self.normalization_enabled = False
        self.pad = False
        self.normalizer = None
        self.composedTransforms = None   
        
        data_root_suffix = os.path.join(self.data_root, self.suffix, type_)
        if not os.path.exists(data_root_suffix):
            os.makedirs(data_root_suffix)

        self.dataset = torchvision.datasets.ImageFolder(data_root_suffix, transform=transforms.Compose(self.getTransforms()), target_transform=None)
        self.mapFileNameToIndex = {} # This dictionary will make it easy to find the information of an image by its file name.
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = dataset_normalization(data_root_suffix, self.dataset, res=params['img_res']).getTransform()[0]
        self.RGBmean = [round(self.normalizer.mean[0]*255), round(self.normalizer.mean[1]*255), round(self.normalizer.mean[2]*255)]
        self.pad = True

        if color_pca is not None:
            self.color_pca = color_pca
        else: 
            self.color_pca = Color_PCA(data_root_suffix, torch.utils.data.DataLoader(self.dataset, batch_size=128), 1.5)

        self.augmentation_enabled = params["augmented"]
        self.normalization_enabled = True
       
        if csv_processor is None:
            self.csv_processor = CSV_processor(self.data_root, self.suffix)
        else:
            self.csv_processor = csv_processor



    def getTransforms(self):
        transformsList = [
            transforms.Lambda(self.MakeSquared)]

        if self.augmentation_enabled:
            transformsList + [transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(degrees=60, translate=(0.25, 0.25), fillcolor=self.RGBmean)]
        
        transformsList = transformsList + [transforms.ToTensor()]

        if self.augmentation_enabled:
            transformsList = transformsList + [transforms.Lambda(self.color_pca.perturb_color)]
              
        if self.normalization_enabled:
            transformsList = transformsList + [self.normalizer]
        return transformsList
       

    def __len__(self):
        return len(self.dataset)

    
    # Toggles whether loaded images are normalized, squared, or augmented
    def toggle_image_loading(self, augmentation=None, normalization=None, pad=None):
        old = (self.augmentation_enabled, self.normalization_enabled, self.pad)
        self.augmentation_enabled = augmentation if augmentation is not None else self.augmentation_enabled
        self.normalization_enabled = normalization if normalization is not None else self.normalization_enabled
        self.pad = pad if pad is not None else self.pad
        self.composedTransforms = None
        return old

    # Makes the image squared while still preserving the aspect ratio
    def MakeSquared(self, img):
        # if imageDimension is None, resize to 224 which works for pretrained ResNet.
        imageDimension = 224 if self.imageDimension is None else self.imageDimension

        img_H = img.size[0]
        img_W = img.size[1]

        # Resize and pad
        smaller_dimension = 0 if img_H < img_W else 1
        larger_dimension = 1 if img_H < img_W else 0
        if self.pad and (imageDimension != img_H or imageDimension != img_W):
            new_smaller_dimension = int(imageDimension * img.size[smaller_dimension] / img.size[larger_dimension])
            if smaller_dimension == 1:
                img = transforms.functional.resize(img, (new_smaller_dimension, imageDimension))
            else:
                img = transforms.functional.resize(img, (imageDimension, new_smaller_dimension))

            diff = imageDimension - new_smaller_dimension
            pad_1 = int(diff/2)
            pad_2 = diff - pad_1
            fill = tuple(self.RGBmean)

            if smaller_dimension == 0:
                img = transforms.functional.pad(img, (pad_1, 0, pad_2, 0), padding_mode='constant', fill = fill)
            else:
                img = transforms.functional.pad(img, (0, pad_1, 0, pad_2), padding_mode='constant', fill = fill)

        return img

    def getIdxByFileName(self, fileName):
        return self.mapFileNameToIndex[fileName]

    def __getitem__(self, idx):   
        if self.composedTransforms is None:
            self.composedTransforms = transforms.Compose(self.getTransforms())
            self.dataset.transform = self.composedTransforms 

        image, target = self.dataset[idx]
        image = image.type(torch.FloatTensor)
        fileName_full = self.dataset.samples[idx][0]
        fileName = os.path.basename(fileName_full)

        if fileName not in self.mapFileNameToIndex.keys():
            self.mapFileNameToIndex[fileName] = idx

        img_fine_label = self.csv_processor.getFineLabel(fileName)
        img_fine_index = self.csv_processor.getFineList().index(img_fine_label)
        assert(target == img_fine_index), f"label in dataset is not the same as file name in csv, {fileName}, in dataset={target}, in csv={img_fine_index}"
        img_fine_index = torch.tensor(img_fine_index)

        matchcoarse = self.csv_processor.getCoarseLabel(fileName)
        matchcoarse_index = torch.tensor(self.csv_processor.getCoarseList().index(matchcoarse))

        return {'image': image, 
                'fine': img_fine_index, 
                'fileName': fileName,
                'fileNameFull': fileName_full,
                'coarse': matchcoarse_index,} 


def getParams(params):
    data_root = params["image_path"]
    suffix = str(params["suffix"]) if ("suffix" in params and params["suffix"] is not None) else ""    
    return data_root, suffix


################################
#### Public class

# This class provides the necessary dataloaders for the experiments
class datasetManager:
    def __init__(self, experimentName, data_path, verbose=False):
        self.verbose = verbose
        self.suffix = None
        self.data_root = None
        self.experimentName = experimentName
        self.datasetName = None
        self.params = None
        self.reset()
    
    def reset(self):
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.train_loader = None
        self.validation_loader =  None
        self.test_loader = None
        
    
    def updateParams(self, params):
        self.params = params

        datasetName = getDatasetName(params)
        if datasetName != self.datasetName:
            self.datasetName = datasetName
            self.data_root, self.suffix = getParams(params)
            self.experiment_folder_name = os.path.join(self.data_root, self.suffix, self.experimentName)
            self.dataset_folder_name = os.path.join(self.experiment_folder_name, datasetName)
            self.reset()
        
    def getDataset(self):
        if self.dataset_train is None:
            print("Creating datasets...")
            self.dataset_train = FishDataset("train", self.params,  None, None, None)
            self.dataset_val = FishDataset("val", self.params, self.dataset_train.normalizer, self.dataset_train.color_pca, self.dataset_train.csv_processor)
            self.dataset_test = FishDataset("test", self.params, self.dataset_train.normalizer, self.dataset_train.color_pca, self.dataset_train.csv_processor)
            print("Creating datasets... Done.")
        return self.dataset_train, self.dataset_val, self.dataset_test

    # Creates the train/val/test dataloaders out of the dataset 
    def getLoaders(self):
        batchSize = self.params["batchSize"]

        if self.dataset_train is None:
            self.getDataset()

        # create data loaders.
        print("Creating loaders...")
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, pin_memory=True, shuffle=True, batch_size=batchSize, num_workers=num_of_workers)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset_val, pin_memory=True, shuffle=True, batch_size=batchSize, num_workers=num_of_workers)
        self.validation_loader.dataset.toggle_image_loading(augmentation=False, normalization=self.dataset_val.normalization_enabled)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, pin_memory=True, shuffle=True, batch_size=batchSize, num_workers=num_of_workers)
        self.test_loader.dataset.toggle_image_loading(augmentation=False, normalization=self.dataset_test.normalization_enabled) # Needed so we always get the same prediction accuracy 
        print("Creating loaders... Done.")
        
        return self.train_loader, self.validation_loader, self.test_loader
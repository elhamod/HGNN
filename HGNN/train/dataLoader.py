import os
from torch.utils.data import Dataset
import numpy as np
<<<<<<< HEAD
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import time
from torchvision import transforms
from tqdm import tqdm
# import joblib
import copy
import random
import csv
import json
from sklearn.model_selection import train_test_split
# from PIL import Image, ImageStat
import hashlib
from random import randrange

from .configParser import getDatasetName, getDatasetParams
from .CSV_processor import CSV_processor


testIndexFileName = "testIndex.csv"
valIndexFileName = "valIndex.csv"
trainingIndexFileName = "trainingIndex.csv"
paramsFileName="params.json"



class FishDataset(Dataset):
    def __init__(self, params, verbose=False):
        self.transformedSamples = {} # caches the transformed samples to speed training up
        self.imageIndicesPerfine = {} # A hash map for fast retreival
        self.imageDimension = 224
        self.n_channels = 3
        self.data_root, self.suffix  = getParams(params)
        self.augmentation_enabled = params["augmented"]
        self.normalization_enabled = True
        self.normalizer = None
        self.composedTransforms = None
        self.fineTocoarseMatrix = None # Maps a vector of fine to coarse.
        
        data_root_suffix = os.path.join(self.data_root, self.suffix)
        if not os.path.exists(data_root_suffix):
            os.makedirs(data_root_suffix)
        

        self.csv_processor = CSV_processor(self.data_root, self.suffix)
        
        # Create transfroms
        if self.normalizer is None:
            self.normalizer = [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]
    
    def getTransforms(self):
        transformsList = [#transforms.ToPILImage(),
              transforms.Lambda(self.MakeSquared),
              transforms.ToTensor()]
              
        if self.normalization_enabled:
            transformsList = transformsList + self.normalizer
=======
import torch
from torchvision import transforms
from tqdm import tqdm
import random
import torchvision
import re

from myhelpers.dataset_normalization import dataset_normalization
from myhelpers.color_PCA import Color_PCA

from .configParser import getDatasetName
from .CSV_processor import CSV_processor



num_of_workers = 8

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def is_valid_file_no_augmentation(path):
    fileName = os.path.basename(path)
    # A file is not valid if it has "XXXX_aug_n.XXX", where n is not 0
    isValid = ("_aug_0." in fileName) or not ("_aug_" in fileName)
    return isValid

class FishDataset(Dataset):
    def __init__(self, type_, params, data_path, normalizer, color_pca, csv_processor, verbose=False):
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
        # Only valid files are the non-augmented ones 
        self.dataset = torchvision.datasets.ImageFolder(data_root_suffix, is_valid_file=is_valid_file_no_augmentation, transform=transforms.Compose(self.getTransforms()), target_transform=None)
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
        transformsList = [#transforms.ToPILImage(),
            transforms.Lambda(self.MakeSquared)]

        if self.augmentation_enabled:
            transformsList + [transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(degrees=60, translate=(0.25, 0.25), fillcolor=self.RGBmean)]
        
        transformsList = transformsList + [transforms.ToTensor()]

        if self.augmentation_enabled:
            transformsList = transformsList + [transforms.Lambda(self.color_pca.perturb_color)]
              
        if self.normalization_enabled:
            transformsList = transformsList + [self.normalizer]
>>>>>>> Loss-surface
        return transformsList
       

    def __len__(self):
<<<<<<< HEAD
        return len(self.csv_processor.samples)

    
    # Toggles whether loaded images are normalized. augmentation is for future use.
    def toggle_image_loading(self, augmentation, normalization):
        old = (self.augmentation_enabled, self.normalization_enabled)
        self.augmentation_enabled = augmentation
        self.normalization_enabled = normalization
        self.composedTransforms = None
        return old
    
    def createTransformedSamples(self, transformHash):
        self.transformedSamples[transformHash] = {}
        with tqdm(total=len(self.csv_processor.samples), desc="Transforming images") as bar:

            # Go through the original images
            for i, sample in enumerate(self.csv_processor.samples):
                images = self.csv_processor.samples[i]['images']
                self.transformedSamples[transformHash][i] = {}

                # For each original, get the first one. Only get the rest if augmentation is enabled
                for j, image in enumerate(images): 
                    image = self.composedTransforms(image)
                    self.transformedSamples[transformHash][i][j] = image
                    if not self.augmentation_enabled:
                        break
                bar.update()

    # Makes the image squared while still preserving the aspect ratio
    def MakeSquared(self, img):
        img_H = img.size[0]
        img_W = img.size[1]

        # Resize
        smaller_dimension = 0 if img_H < img_W else 1
        larger_dimension = 1 if img_H < img_W else 0
        new_smaller_dimension = int(self.imageDimension * img.size[smaller_dimension] / img.size[larger_dimension])
        if smaller_dimension == 1:
            img = transforms.functional.resize(img, (new_smaller_dimension, self.imageDimension))
        else:
            img = transforms.functional.resize(img, (self.imageDimension, new_smaller_dimension))

        # pad
        diff = self.imageDimension - new_smaller_dimension
        pad_1 = int(diff/2)
        pad_2 = diff - pad_1
    #         if self.normalizeFromResnet:
        RGBmean = [0.485*255, 0.456*255, 0.406*255]
        fill = tuple([round(x) for x in RGBmean])
    #         else:
    #             stat = ImageStat.Stat(img)
    #             fill = tuple([round(x) for x in stat.mean])

        if smaller_dimension == 0:
            img = transforms.functional.pad(img, (pad_1, 0, pad_2, 0), padding_mode='constant', fill = fill)
        else:
            img = transforms.functional.pad(img, (0, pad_1, 0, pad_2), padding_mode='constant', fill = fill)

        return img
    
    
    def __getitem__(self, idx):          
        img_fine = self.csv_processor.samples[idx]['fine']
        img_fine_index = torch.tensor(self.csv_processor.getFineList().index(img_fine))
        
        # Cache transformed images
        if self.composedTransforms is None:
            self.composedTransforms = transforms.Compose(self.getTransforms())
        hashString = str(self.augmentation_enabled) + str(self.normalization_enabled)
        transformHash = hashlib.sha224(hashString.encode('utf-8')).hexdigest()
            
        if transformHash not in self.transformedSamples:
            self.createTransformedSamples(transformHash)

        imageList = self.transformedSamples[transformHash][idx]
        numOfImages = len(imageList)
        image = imageList[randrange(numOfImages)]     
               
        fileName = self.csv_processor.samples[idx]['fileName']
#         matchFamily = self.csv_processor.samples[idx]['family']
        matchcoarse = self.csv_processor.samples[idx]['coarse']
        matchcoarse_index = torch.tensor(self.csv_processor.getCoarseList().index(matchcoarse))
    
        if torch.cuda.is_available():
            image = image.cuda()
            matchcoarse_index = matchcoarse_index.cuda()
            img_fine_index = img_fine_index.cuda()
=======
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
        fileName = re.sub(r'_{}_aug_\d+'.format(self.imageDimension), '', fileName)

        if fileName not in self.mapFileNameToIndex.keys():
            self.mapFileNameToIndex[fileName] = idx

        img_fine_label = self.csv_processor.getFineLabel(fileName)
        img_fine_index = self.csv_processor.getFineList().index(img_fine_label)
        assert(target == img_fine_index), f"label in dataset is not the same as file name in csv, {fileName}, in dataset={target}, in csv={img_fine_index}"
        img_fine_index = torch.tensor(img_fine_index)

#         matchFamily = self.csv_processor.samples[idx]['family']
        matchcoarse = self.csv_processor.getCoarseLabel(fileName)
        matchcoarse_index = torch.tensor(self.csv_processor.getCoarseList().index(matchcoarse))
>>>>>>> Loss-surface

        return {'image': image, 
                'fine': img_fine_index, 
                'fileName': fileName,
<<<<<<< HEAD
                'coarse': matchcoarse_index,} 
    
    
    
    
    
    

def writeFile(folder_name, file_name, obj):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    try:
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(obj)
#         with open(file_name, 'wb') as f:
# #             joblib.dump(obj, f) 
        print('file',file_name,'written')
    except:
        print("Couldn't write", file_name)
        pass
        
def readFile(fullFileName):
    try:
        with open(fullFileName, newline='') as f:
            reader = csv.reader(f)
            loaded = list(reader)
#         with open(fullFileName, 'rb') as filehandle:
# #             loaded = joblib.load(filehandle) 
            print('file',fullFileName,'read')
            loaded[0] = [int(i) for i in loaded[0]]
            return loaded[0]
    except:
        print("Couldn't read", fullFileName)
        pass  

    
    
    
    
=======
                'fileNameFull': fileName_full,
                'coarse': matchcoarse_index,} 

>>>>>>> Loss-surface

def getParams(params):
    data_root = params["image_path"]
    suffix = str(params["suffix"]) if ("suffix" in params and params["suffix"] is not None) else ""    
    return data_root, suffix
    
class datasetManager:
<<<<<<< HEAD
    def __init__(self, experimentName, verbose=False):
=======
    def __init__(self, experimentName, data_path, verbose=False):
>>>>>>> Loss-surface
        self.verbose = verbose
        self.suffix = None
        self.data_root = None
        self.experimentName = experimentName
        self.datasetName = None
<<<<<<< HEAD
        self.loader_indices = []
        self.params = None
        self.reset()
    
    def reset(self):
        self.dataset = None
=======
        self.params = None
        self.data_path = data_path
        self.reset()
    
    def reset(self):
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
>>>>>>> Loss-surface
        self.train_loader = None
        self.validation_loader =  None
        self.test_loader = None
        
    
    def updateParams(self, params):
<<<<<<< HEAD
        params_copy = copy.copy(params)
        self_params_copy = copy.copy(self.params)

        datasetName = getDatasetName(params)
        if datasetName != self.datasetName:
            self.reset()
            self.params = params
            self.data_root, self.suffix = getParams(params)
            self.experiment_folder_name = os.path.join(self.data_root, self.suffix, self.experimentName)
            self.dataset_folder_name = os.path.join(self.experiment_folder_name, datasetName)
            self.datasetName = datasetName


        # except for augmentation, if dataset has changed, reset the indices     

        if self_params_copy is not None:
            self_params_copy["augmented"] = None 
        params_copy["augmented"] = None 
        if (self_params_copy is None) or (getDatasetName(params_copy) != getDatasetName(self_params_copy)):
            self.loader_indices = []
        
    def getDataset(self):
        if self.dataset is None:
            print("Creating dataset...")
            self.dataset = FishDataset(self.params, self.verbose)
            print("Creating dataset... Done.")
        return self.dataset

    # Creates the train/val/test dataloaders out of the dataset 
    def getLoaders(self, resplit=False):
        index_fileNames = [trainingIndexFileName, valIndexFileName, testIndexFileName]
        saved_index_file = os.path.join(self.dataset_folder_name, testIndexFileName)
        batchSize = self.params["batchSize"]

        if resplit or self.loader_indices == []:
            if self.dataset is None:
                self.getDataset()

            training_count = self.params["training_count"]        
            validation_count = self.params["validation_count"]
           
 
            
            if not os.path.exists(saved_index_file):
                
                indices_len = len(self.dataset)
                data_loader = torch.utils.data.DataLoader(self.dataset,
                                            batch_size=indices_len,
                                            shuffle=False)
                data_loader_batch = next(iter(data_loader))
                indices = range(indices_len)
                labels = data_loader_batch['fine']
                if torch.cuda.is_available():
                    labels = labels.cpu() 
                train_indices, test_indices = train_test_split(indices, test_size= 1-training_count-validation_count, 
                                                            stratify=labels.cpu())

                print("train/test = ", len(train_indices),len(test_indices))
                data_loader = torch.utils.data.DataLoader(self.dataset,
                                            batch_size=indices_len,
                                            shuffle=False)
                data_loader_batch = next(iter(data_loader))
                labels_sub = data_loader_batch['fine'][train_indices]
                if torch.cuda.is_available():
                    labels_sub = labels_sub.cpu() 
                train_indices, val_indices = train_test_split(train_indices, test_size= validation_count/(validation_count+training_count), 
                                                            stratify=labels_sub.cpu())
        
                print("train/val = ", len(train_indices),len(val_indices))
                self.loader_indices = [train_indices, val_indices, test_indices]

            else:
                # load the pickles
                print("Loading saved indices...")
                for i, name in enumerate(index_fileNames):   
                    f = readFile( os.path.join(self.dataset_folder_name, name))
                    self.loader_indices.append(f)
                    

        # create samplers
        train_sampler = SubsetRandomSampler(self.loader_indices[0])
        valid_sampler = SubsetRandomSampler(self.loader_indices[1])
        test_sampler = SubsetRandomSampler(self.loader_indices[2])

        # create data loaders.
        print("Creating loaders...")
        self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=train_sampler, batch_size=batchSize)
        self.validation_loader = torch.utils.data.DataLoader(copy.copy(self.dataset), sampler=valid_sampler, batch_size=batchSize)
        self.validation_loader.dataset.toggle_image_loading(augmentation=False, normalization=self.dataset.normalization_enabled)
        self.test_loader = torch.utils.data.DataLoader(copy.copy(self.dataset), sampler=test_sampler, batch_size=batchSize)
        self.test_loader.dataset.toggle_image_loading(augmentation=False, normalization=self.dataset.normalization_enabled) # Needed so we always get the same prediction accuracy 
        print("Creating loaders... Done.")
            
        
        if not os.path.exists(saved_index_file):
            # save indices
            for i, name in enumerate(index_fileNames):
                fullFileName = os.path.join(self.dataset_folder_name, name)
                writeFile(self.dataset_folder_name, fullFileName, self.loader_indices[i])
            # save params
            j = json.dumps(getDatasetParams(self.params))
            f = open(os.path.join(self.dataset_folder_name, paramsFileName),"w")        
            f.write(j)
            f.close()   

=======
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
            self.dataset_train = FishDataset("train", self.params, self.data_path, None, None, None, self.verbose)
            self.dataset_val = FishDataset("val", self.params, self.data_path, self.dataset_train.normalizer, self.dataset_train.color_pca, self.dataset_train.csv_processor, self.verbose)
            self.dataset_test = FishDataset("test", self.params, self.data_path, self.dataset_train.normalizer, self.dataset_train.color_pca, self.dataset_train.csv_processor, self.verbose)
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
>>>>>>> Loss-surface
        
        return self.train_loader, self.validation_loader, self.test_loader
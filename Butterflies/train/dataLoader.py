# kfold reference: https://datascience.stackexchange.com/questions/49571/can-we-use-k-fold-cross-validation-without-any-extra-excluded-test-set 

import os
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import random
import torchvision
import re
from sklearn.model_selection import StratifiedKFold

from myhelpers.dataset_normalization import dataset_normalization
from myhelpers.color_PCA import Color_PCA
from myhelpers.seeding import get_seed_from_trialNumber
from myhelpers.read_write import Pickle_reader_writer
from myhelpers.imbalanced import ImbalancedDatasetSampler


from .configParser import getDatasetName
from .CSV_processor import CSV_processor



num_of_workers = 8
kf_splits_FILENAME = "kf_splits.pkl"
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def is_valid_file_no_augmentation(path):
    fileName = os.path.basename(path)
    # A file is not valid if it has "XXXX_aug_n.XXX", where n is not 0
    isValid = ("_aug_0." in fileName) or not ("_aug_" in fileName)
    return isValid


SHUFFLE=False # no need to shuffle the batches every epoch
DROPLAST = False # False is not good for randomness, but True (dropping examples) might hurt performance!

def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
        self.grayscale = params["grayscale"]
        self.random_fitting = params["random_fitting"]
        self.ring_as_coarse = params["coarseType"]
        
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
            self.color_pca = Color_PCA(data_root_suffix, torch.utils.data.DataLoader(self.dataset, batch_size=128, num_workers=num_of_workers, drop_last=DROPLAST, worker_init_fn=_init_fn), 1.5)

        self.augmentation_enabled = params["augmented"]
        self.normalization_enabled = True
       
        
        if csv_processor is None:
            self.csv_processor = CSV_processor(self.data_root, self.suffix, build_taxonomy=False)
        else:
            self.csv_processor = csv_processor
            
        # In case we want some stats
#         label_and_freq = torch.unique(torch.tensor(self.dataset.targets), return_counts=True)
#         for i in self.dataset.classes:
#             print(i, self.dataset.class_to_idx[i], label_and_freq[1][self.dataset.class_to_idx[i]].cpu().detach().numpy())
#         raise

    def getTransforms(self):
        transformsList = [#transforms.ToPILImage(),
            transforms.Lambda(self.MakeSquared)]

        if self.augmentation_enabled:
            transformsList + [transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(degrees=60, translate=(0.25, 0.25), fill=self.RGBmean)]
        
        if self.grayscale:
            transformsList = transformsList + [transforms.Grayscale(num_output_channels=1)]

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

    def get_labels(self):
        return [x[1] for x in self.dataset.imgs]
    
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
        if self.ring_as_coarse=="coarse":
            matchcoarse = self.csv_processor.getCoarseLabel(fileName)
            matchcoarse_index = torch.tensor(self.csv_processor.getCoarseList().index(matchcoarse))
        else:
            matchcoarse_index = torch.tensor(self.csv_processor.comimics_components[img_fine_index])

        return {'image': image, 
                'fine': img_fine_index if not self.random_fitting else hash(fileName_full)%len(self.csv_processor.getFineList()), 
                'fileName': fileName,
                'fileNameFull': fileName_full,
                'coarse': matchcoarse_index,} 


def getParams(params):
    data_root = params["image_path"]
    suffix = str(params["suffix"]) if ("suffix" in params and params["suffix"] is not None) else ""    
    return data_root, suffix
    
class datasetManager:
    def __init__(self, experimentName, data_path, verbose=False):
        self.verbose = verbose
        self.suffix = None
        self.data_root = None
        self.experimentName = experimentName
        self.datasetName = None
        self.params = None
        self.data_path = data_path
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
        useCrossValidation = self.params["useCrossValidation"]
        if self.dataset_train is None:
            print("Creating datasets...")
            self.dataset_train = FishDataset("train", self.params, self.data_path, None, None, None, self.verbose)

            # if useCrossValidation and "val" folder exists, "train" and "val" will be joined
            # if useCrossValidation and "val" does not folder exists, "train" will be used
            # if not useCrossValidation and "val" folder exists, use "train" and "val" separately
            # if not useCrossValidation and "val" folder does not exist, raise an error!
            self.dataset_val = None
            try:
                self.dataset_val = FishDataset("val", self.params, self.data_path, self.dataset_train.normalizer, self.dataset_train.color_pca, self.dataset_train.csv_processor, self.verbose)
            except:
                if not useCrossValidation:
                    print("Either cross-validation need to be enabled or a 'val' DatasetFolder needs to exist")
                    raise
            
            self.dataset_test = FishDataset("test", self.params, self.data_path, self.dataset_train.normalizer, self.dataset_train.color_pca, self.dataset_train.csv_processor, self.verbose)
            print("Creating datasets... Done.")
        return self.dataset_train, self.dataset_val, self.dataset_test

    # Creates the train/val/test dataloaders out of the dataset 
    def getLoaders(self, trial_num=None):
        batchSize = self.params["batchSize"]
        useCrossValidation = self.params["useCrossValidation"]
        n_splits = self.params["numOfTrials"]
        useImbalancedSampling = self.params["useImbalancedSampling"]

        SEED_INT = get_seed_from_trialNumber(trial_num)

        if self.dataset_train is None:
            self.getDataset()

            if useCrossValidation:
                # Try to load saved kf_splits. If they don't exist, generate them
                split_filepath = os.path.join(self.data_root, self.suffix)
                split_filename = str(n_splits) + "_" + kf_splits_FILENAME
                read_writer = Pickle_reader_writer(split_filepath, split_filename)
                self.kf_splits = read_writer.readFile()

                targets = self.dataset_train.dataset.targets
                if self.dataset_val is not None:
                    self.dataset_train = torch.utils.data.ConcatDataset([self.dataset_train,self.dataset_val])
                    self.dataset_train.csv_processor = self.dataset_val.csv_processor
                    targets = targets + self.dataset_train.datasets[1].dataset.targets

                if self.kf_splits is None:
                    self.kf_splits = StratifiedKFold(n_splits=n_splits, random_state=SEED_INT).split(self.dataset_train, targets)
                    self.kf_splits = [(train_index, val_index) for train_index, val_index in self.kf_splits]

                    # Save the splits
                    read_writer.writeFile(self.kf_splits)
            
        # create data loaders.
        print("Creating loaders...")
        train_generator = torch.Generator()
        train_generator.manual_seed(SEED_INT)
        val_generator = torch.Generator()
        val_generator.manual_seed(SEED_INT)
        # train_generator = None
        # val_generator = None
        if not useCrossValidation:
            train_subsampler = None
            if useImbalancedSampling:
                train_subsampler = ImbalancedDatasetSampler(self.dataset_train)
            self.train_loader = torch.utils.data.DataLoader(self.dataset_train, pin_memory=True, generator=train_generator, sampler=train_subsampler, batch_size=batchSize, num_workers=num_of_workers, drop_last=DROPLAST, worker_init_fn=_init_fn)
            self.validation_loader = torch.utils.data.DataLoader(self.dataset_val, pin_memory=True, generator=val_generator, shuffle=SHUFFLE, batch_size=batchSize, num_workers=num_of_workers, drop_last=DROPLAST, worker_init_fn=_init_fn)
        elif trial_num is not None:
            # train_index, val_index = next(self.kf_splits)
            (train_index, val_index) = self.kf_splits[trial_num]
            # print(trial_num, train_index, val_index, len(train_index), len(val_index))
            if not useImbalancedSampling:
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_index) #SubsetRandomSampler
            else:
                train_subsampler = ImbalancedDatasetSampler(self.dataset_train, indices=train_index)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_index) # SubsetRandomSampler
            self.train_loader = torch.utils.data.DataLoader(self.dataset_train, pin_memory=True, generator=train_generator, sampler=train_subsampler, batch_size=batchSize, num_workers=num_of_workers, drop_last=DROPLAST, worker_init_fn=_init_fn)
            self.validation_loader = torch.utils.data.DataLoader(self.dataset_train, pin_memory=True, generator=val_generator, sampler=val_subsampler, batch_size=batchSize, num_workers=num_of_workers, drop_last=DROPLAST, worker_init_fn=_init_fn)
        else:
            print("getLoaders with useCrossValidation should specify a fold (trial_num) number!")
            raise
        
        # TODO: Had to remove this because with crossvalidation .dataset has a different meaning.
        # It is OK for now but it changes behavior.
        # self.validation_loader.dataset.toggle_image_loading(augmentation=False, normalization=self.dataset_val.normalization_enabled)
        
        test_generator = torch.Generator()
        test_generator.manual_seed(SEED_INT)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, pin_memory=True, generator=test_generator, shuffle=SHUFFLE, batch_size=batchSize, num_workers=num_of_workers, drop_last=DROPLAST, worker_init_fn=_init_fn)
        self.test_loader.dataset.toggle_image_loading(augmentation=False, normalization=self.dataset_test.normalization_enabled) # Needed so we always get the same prediction accuracy 
        print("Creating loaders... Done.")

        return self.train_loader, self.validation_loader, self.test_loader
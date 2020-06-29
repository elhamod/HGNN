import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torchvision import transforms, datasets
import pandas as pd
import progressbar
import joblib
from configParser import getDatasetName
import copy
import torchvision.transforms.functional as TF
import random

testIndexFileName = "testIndex.pkl"
valIndexFileName = "valIndex.pkl"
trainingIndexFileName = "trainingIndex.pkl"

class Dictlist(dict):
    def __setitem__(self, key, value):
        try:
            self[key]
        except KeyError:
            super(Dictlist, self).__setitem__(key, [])
        self[key].append(value)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
        
class CifarDataset(Dataset):
    def __init__(self, params, training=True, verbose=False):
        self.samples = [] # The list of all samples
        self.transformedSamples = {} # caches the transformed samples to speed training up
        self.imageIndicesPerSpecies = {} # A hash map for fast retreival
        self.data_root, self.suffix  = getParams(params)
        self.augmentation_enabled = False
        self.normalization_enabled = True
        self.normalizeFromResnet = params["normalizeFromResnet"] if ("normalizeFromResnet" in params) else False
        self.speciesList= None
        self.genusList= None
        self.normalizer = None
        self.transforms = None
        self.composedTransforms = None
        self.speciesToGenusMatrix = None
        resnet = params["resnet"]
        
        data_root_suffix = os.path.join(self.data_root, self.suffix)
        if not os.path.exists(data_root_suffix):
            os.makedirs(data_root_suffix)     
            
        print("Loading dataset...")            
        self.dataset = datasets.CIFAR100(data_root_suffix, download=True, train=training, target_transform=self.getTargetTransform)

        x=unpickle(os.path.join(self.data_root,'cifar-100-python/train'))
        self.coarse_to_fine=Dictlist()
        for i in range(0,len(x[b'coarse_labels'])):
            self.coarse_to_fine[x[b'coarse_labels'][i]]=x[ b'fine_labels'][i]
        self.coarse_to_fine=dict(self.coarse_to_fine)
        for i in self.coarse_to_fine.keys():
            self.coarse_to_fine[i]=list(dict.fromkeys(self.coarse_to_fine[i]))
        self.fileNames = x[b'filenames']
            
        metadata = unpickle(os.path.join(self.data_root,'cifar-100-python/meta'))
        self.coarse_index_list = metadata[b'coarse_label_names']

        # Create transfroms
        # Toggle beforehand so we could create the normalization transform. Then toggle back.
        if self.normalizer is None:
            augmentation, normalization = self.toggle_image_loading(augmentation=False, normalization=False)   
            print("CIFAR normalization")
            # Cifar normalization: 
            self.normalizer = [transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
            self.toggle_image_loading(augmentation, normalization)
    
    def getTransforms(self):
        transformsList = [transforms.ToTensor()]
            
        if self.normalization_enabled:
            transformsList = transformsList + self.normalizer
        
        return transformsList

    def __len__(self):
        return len(self.dataset)
    
    # The list of species/genus names
    def getSpeciesList(self):
        return self.dataset.classes
    def getGenusList(self):
        return self.coarse_index_list
    
    def getNumberOfImagesForSpecies(self, species):
        return len([item for item in self.dataset.targets if self.dataset.classes[item] == species])
    
    # returns the indices of a species in self.samples
    def getSpeciesIndices(self, species):
        return [i for i in range(len(self.dataset.targets)) if self.dataset.targets[i] == self.dataset.classes.index(species)]
    
    # Convert index to species name.
    def getSpeciesOfIndex(self, index):
        return self.getSpeciesList()[index]
    
    def getGenusFromSpecies(self, species):
        speciesIndex = self.dataset.classes.index(species)
        for coarse in self.coarse_to_fine:
            speciesInGenus = self.coarse_to_fine[coarse]
            if speciesIndex in speciesInGenus:
                return self.coarse_index_list[coarse]
    
    # Returns a list of species_index that belong to a genus
    def getSpeciesWithinGenus(self, genus):
        return list(map(lambda x: self.dataset.classes[x], self.coarse_to_fine[self.coarse_index_list.index(genus)]))
    

    def getSpeciesToGenusMatrix(self):
        if self.speciesToGenusMatrix is None:
            self.speciesToGenusMatrix = torch.zeros(len(self.getSpeciesList()), len(self.getGenusList()))
            for species_name in self.getSpeciesList():
                genus_name = self.getGenusFromSpecies(species_name)
                species_index = self.getSpeciesList().index(species_name)
                genus_index = self.getGenusList().index(genus_name)
                self.speciesToGenusMatrix[species_index][genus_index] = 1
        return self.speciesToGenusMatrix
    
    def toggle_image_loading(self, augmentation, normalization):
        old = (self.augmentation_enabled, self.normalization_enabled)
        self.augmentation_enabled = augmentation
        self.normalization_enabled = normalization
        self.transforms = None
        return old
        
    def getTargetTransform(self, target):
        return {
            'species': target,
            'genus': self.coarse_index_list.index(self.getGenusFromSpecies(self.dataset.classes[target]))
        }

    def __getitem__(self, idx):       
        if self.transforms is None:
            self.transforms = self.getTransforms()
            self.composedTransforms = transforms.Compose(self.transforms)
            self.dataset.transform = self.composedTransforms
            
        image, target = self.dataset[idx]
        if torch.cuda.is_available():
            image = image.cuda()

        return {'image': image, 
                'species': target['species'], 
                'fileName': self.fileNames[idx],
                'genus': target['genus'],} 
    

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
def getExamples(model, dataset, trueIndex, expectedIndex, key="species", num_of_examples=1):
    result = []
    
    # get all examples trueLabel=speciesIndex
    if key == "species":
        name = dataset.getSpeciesOfIndex(trueIndex)
        examples = dataset.getSpeciesIndices(name)
    else:
        examples = []
        species_set = dataset.getSpeciesWithinGenus(dataset.getGenusList()[trueIndex])
        for i in species_set:
            examples = examples + dataset.getSpeciesIndices(i)  
            

    # Find an example that predictedLabel=expectedIndex
#     random.shuffle(examples)
    for example in examples:
        augmentation, normalization = dataset.toggle_image_loading(augmentation=False, normalization=False)
        image = dataset[example]['image'].unsqueeze(0)
        dataset.toggle_image_loading(augmentation, normalization)
        predictionImage = dataset[example]['image'].unsqueeze(0)
        predictedIndex = model(predictionImage)
        if isinstance(predictedIndex,dict):
            predictedIndex = predictedIndex[key]
        predictedIndex = torch.max(predictedIndex.data, 1)[1].cpu().detach().numpy()[0]
        if (predictedIndex == expectedIndex):
            image = image.squeeze()
            predictionImage = predictionImage.squeeze()
            result.append((image, predictionImage))
            if len(result) == num_of_examples:
                break

    return result



def getIndices(data_root, train=True):
    fileName = 'train_train.txt' if train else 'train_val.txt'
    read_file = pd.read_csv(os.path.join(data_root,fileName), header = None, delimiter=' ')
    read_file.columns = ['index','target']
    return read_file['index'].tolist()

def getParams(params):
    data_root = params["image_path"]
    suffix = str(params["suffix"]) if ("suffix" in params and params["suffix"] is not None) else ""    
    return data_root, suffix
    
class datasetManager:
    def __init__(self, experimentName, verbose=False):
        self.verbose = verbose
        self.suffix = None
        self.dataset_train = None
        self.dataset_test = None
        self.experimentName = experimentName
        self.reset()
    
    def reset(self):
        self.dataset_train = None
        self.dataset_test = None
        self.train_loader = None
        self.validation_loader =  None
        self.test_loader = None
    
    def updateParams(self, params):
        self.reset()
        self.params = params
        self.data_root, self.suffix = getParams(params)
        self.experiment_folder_name = os.path.join(self.data_root, self.suffix, self.experimentName)
        self.dataset_folder_name = self.experiment_folder_name
        
    def getDataset(self):
        if self.dataset_train is None:
            print("Creating dataset...")
            self.dataset_train = CifarDataset(self.params, verbose=self.verbose, training=True)
            self.dataset_test = CifarDataset(self.params, verbose=self.verbose, training=False)
            print("Creating dataset... Done.")
        return self.dataset_train, self.dataset_test

    # Creates the train/val/test dataloaders out of the dataset 
    def getLoaders(self, readLoadersFromDisk=True):
        if self.dataset_train is None:
            self.getDataset()

#         training_count = self.params["training_count"]        
#         validation_count = self.params["validation_count"]
        batchSize = self.params["batchSize"]

        index_fileNames = [trainingIndexFileName, valIndexFileName]
        saved_index_file = os.path.join(self.dataset_folder_name, valIndexFileName)
        loader_indices = []
        if not os.path.exists(saved_index_file):
            
            train_indices = getIndices(self.data_root)
            val_indices = getIndices(self.data_root, False)

            print("train/val = ", len(train_indices),len(val_indices))
            # save indices
            loader_indices = [train_indices, val_indices]
            for i, name in enumerate(index_fileNames):
                fullFileName = os.path.join(self.dataset_folder_name, name)
                writeFile(self.dataset_folder_name, fullFileName, loader_indices[i])

        else:
            # load the pickles
            print("Loading saved indices...")
            for i, name in enumerate(index_fileNames):        
                loader_indices.append(readFile( os.path.join(self.dataset_folder_name, name)))


        # create samplers
        train_sampler = SubsetRandomSampler(loader_indices[0])
        valid_sampler = SubsetRandomSampler(loader_indices[1])

        # create data loaders.
        print("Creating loaders...")
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, sampler=train_sampler, batch_size=batchSize)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset_train, sampler=valid_sampler, batch_size=batchSize)
        self.test_loader = torch.utils.data.DataLoader(copy.copy(self.dataset_test), batch_size=batchSize)
        self.test_loader.dataset.toggle_image_loading(augmentation=False, normalization=self.dataset_test.normalization_enabled) # Needed so we always get the same prediction accuracy 
        print("Creating loaders... Done.")

            
        return self.train_loader, self.validation_loader, self.test_loader
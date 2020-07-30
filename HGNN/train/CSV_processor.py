import os, glob
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time

# metadata file provided by dataset.
fine_csv_fileName = "metadata.csv"
# cleaned up metadata file that has no duplicates, invalids...etc
cleaned_fine_csv_fileName = "cleaned_metadata.csv"

# Saved file names.
statistic_countPerFine="count_per_fine.csv"
statistic_countPerFamilyAndGenis="count_per_family_genus.csv"

# metadata table headers.
fine_csv_fileName_header = "fileName"
fine_csv_scientificName_header = "scientificName"
fine_csv_Coarse_header = "Genus"
fine_csv_Family_header = "Family"
fine_csv_usedColumns = [fine_csv_fileName_header,
                          fine_csv_scientificName_header,
                          fine_csv_Coarse_header,
                          fine_csv_Family_header]

# subpath of where images can be found.
image_subpath = "images"

# Loads, processes, cleans up and analyise fish metadata
class CSV_processor:
    def __init__(self, data_root, suffix, data_path, params, verbose=False):
        self.data_root = data_root
        self.suffix = suffix
        self.augmentation_enabled = params['augmented']
        self.imageDimension = params['img_res']
        self.image_subpath = image_subpath
        self.fine_csv = None

        self.get_csv_file()
        self.cleanup_csv_file()
        self.save_csv_file()

    def getCoarseLabel(self, fileName):
        return self.fine_csv.loc[fileName][fine_csv_Coarse_header]
    def getFineLabel(self, fileName):
        return self.fine_csv.loc[fileName][fine_csv_scientificName_header]
    
    # The list of fine/coarse names
    def getFineList(self):
         return sorted(self.fine_csv[fine_csv_scientificName_header].unique().tolist())    
    def getCoarseList(self): 
        return sorted(self.fine_csv[fine_csv_Coarse_header].unique().tolist())   
    
    # Fine/Coarse conversions
    def getFineWithinCoarse(self, coarse):
        return self.fine_csv.loc[self.fine_csv[fine_csv_Coarse_header] == coarse][fine_csv_scientificName_header].unique().tolist()
    def getCoarseFromFine(self, fine):
        return self.fine_csv.loc[self.fine_csv[fine_csv_scientificName_header] == fine][fine_csv_Coarse_header].unique().tolist()[0]
    def getFineToCoarseMatrix(self):
        fineToCoarseMatrix = torch.zeros(len(self.getFineList()), len(self.getCoarseList()))
        for fine_name in self.getFineList():
            coarse_name = self.getCoarseFromFine(fine_name)
            fine_index = self.getFineList().index(fine_name)
            coarse_index = self.getCoarseList().index(coarse_name)
            fineToCoarseMatrix[fine_index][coarse_index] = 1
        return fineToCoarseMatrix
    
    def save_csv_file(self):
        cleaned_fine_csv_fileName_full_path = os.path.join(self.data_root, self.suffix, cleaned_fine_csv_fileName)
        # clean up the csv file from unfound images
        if not os.path.exists(cleaned_fine_csv_fileName_full_path):
            self.fine_csv.to_csv(cleaned_fine_csv_fileName_full_path, sep='\t')       
        
    
    # Loads metadata with some cleaning
    def get_csv_file(self):
        # Create fine_csv
        cleaned_fine_csv_fileName_full_path = os.path.join(self.data_root, self.suffix, cleaned_fine_csv_fileName)
        if not os.path.exists(cleaned_fine_csv_fileName_full_path):
            # Load csv file, remove duplicates and invalid.
            csv_full_path = os.path.join(self.data_root, fine_csv_fileName)
            self.fine_csv = pd.read_csv(csv_full_path, delimiter='\t', index_col=fine_csv_fileName_header, usecols=fine_csv_usedColumns)
            self.fine_csv = self.fine_csv.loc[~self.fine_csv.index.duplicated(keep='first')]                         
            self.fine_csv = self.fine_csv[self.fine_csv[fine_csv_Coarse_header] != '#VALUE!']

        else:
            self.fine_csv = pd.read_csv(cleaned_fine_csv_fileName_full_path, delimiter='\t', index_col=fine_csv_fileName_header, usecols=fine_csv_usedColumns) 

        # and sort
        self.fine_csv = self.fine_csv.sort_values(by=[fine_csv_Family_header, fine_csv_Coarse_header])
    
    def get_image_full_path(self):
        return os.path.join(self.data_root, self.image_subpath)
        
    # validates the csv file vs available images
    def cleanup_csv_file(self):
        img_full_path = self.get_image_full_path()

        #get intersection between csv file and list of images
        fileNames1 = os.listdir(img_full_path)
        fileNames2 = self.fine_csv.index.values.tolist()
        fileNames = [value for value in tqdm(fileNames2, desc="scanning files") if value in fileNames1]
        
        self.fine_csv = self.fine_csv.loc[fileNames]
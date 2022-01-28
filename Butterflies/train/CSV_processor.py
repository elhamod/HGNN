import os, glob
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time
from .taxonomy import Taxonomy

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix




# metadata file provided by dataset.
fine_csv_fileName = "metadata.csv"
cleaned_fine_tree_fileName = "cleaned_metadata.tre"
comimics_fileName = 'comimics.csv'

# metadata table headers.
fine_csv_ott_header = 'ott_id'
fine_csv_fileName_header = "Image filename"
fine_csv_scientificName_header = "Fullsubspecies"
fine_csv_Coarse_header = "Species"
fine_csv_usedColumns = [fine_csv_fileName_header,
                          fine_csv_scientificName_header,
                          fine_csv_Coarse_header]
SEP = ","

# subpath of where images can be found.
image_subpath = "images"

# Loads, processes, cleans up and analyise fish metadata
class CSV_processor:
    def __init__(self, data_root, suffix, cleanup=False, verbose=False, cleaned_fine_csv_fileName = "cleaned_metadata.csv", build_taxonomy=True):
        self.data_root = data_root
        self.suffix = suffix
        self.image_subpath = image_subpath
        self.fine_csv = None
        self.cleaned_fine_csv_fileName = cleaned_fine_csv_fileName

        # taxa related
        self.tax = None
        self.distance_matrix = None

        self.get_csv_file()
        if cleanup:
            self.cleanup_csv_file()

        # Building the taxonomy and fixing csv if needed.
        if build_taxonomy:
            self.build_taxonomy()
            if cleanup or (fine_csv_ott_header not in self.fine_csv):
                # add ott_ids if they don't exist
                self.fine_csv[fine_csv_ott_header] = self.fine_csv.apply(lambda row: self.tax.ott_id_dict[row[fine_csv_scientificName_header]], axis=1)

        self.save_csv_file()

        self.comimics_components = None
        self.comimics_components_lables = None

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
        cleaned_fine_csv_fileName_full_path = os.path.join(self.data_root, self.suffix, self.cleaned_fine_csv_fileName)
        # clean up the csv file from unfound images
        if not os.path.exists(cleaned_fine_csv_fileName_full_path):
            self.fine_csv.to_csv(cleaned_fine_csv_fileName_full_path, sep=SEP)

    
    # Loads metadata with some cleaning
    def get_csv_file(self):
        # Create fine_csv
        cleaned_fine_csv_fileName_full_path = os.path.join(self.data_root, self.suffix, self.cleaned_fine_csv_fileName)
        if not os.path.exists(cleaned_fine_csv_fileName_full_path):
            # Load csv file, remove duplicates and invalid.
            csv_full_path = os.path.join(self.data_root, self.suffix, fine_csv_fileName)
            self.fine_csv = pd.read_csv(csv_full_path, delimiter=SEP, index_col=fine_csv_fileName_header, usecols=fine_csv_usedColumns)
            self.fine_csv = self.fine_csv.loc[~self.fine_csv.index.duplicated(keep='first')]                         
            self.fine_csv = self.fine_csv[self.fine_csv[fine_csv_Coarse_header] != '#VALUE!']
            mask = self.fine_csv.index.map(lambda x: isinstance(x, str))
            self.fine_csv = self.fine_csv[mask]
            # strip trailing white spaces
            self.fine_csv = self.fine_csv.applymap(lambda x: x.strip().capitalize())

        else:
            self.fine_csv = pd.read_csv(cleaned_fine_csv_fileName_full_path, delimiter=SEP, index_col=fine_csv_fileName_header, usecols=fine_csv_usedColumns) 

        # and sort
        self.fine_csv = self.fine_csv.sort_values(by=[fine_csv_scientificName_header])
    
    def get_image_full_path(self):
        return os.path.join(self.data_root, self.suffix, self.image_subpath)
        
    # validates the csv file vs available images
    def cleanup_csv_file(self):
        img_full_path = self.get_image_full_path()

        #get intersection between csv file and list of images
        fileNames_dir = os.listdir(img_full_path)

        self.fine_csv.index = self.fine_csv.index.map(lambda x: get_equivalent(x, fileNames_dir))
        mask = self.fine_csv.index.map(lambda x: x is not None)
        self.fine_csv = self.fine_csv[mask]

        print(self.fine_csv)
        # self.fine_csv = self.fine_csv[self.fine_csv.index.isin(fileNames)]

    def build_taxonomy(self):
        df_nodupes = self.fine_csv[fine_csv_scientificName_header].drop_duplicates() # Will probably need more processing to deal with small letter...etc
        node_ids = df_nodupes.tolist()

        cleaned_fine_tree_fileName_full_path = os.path.join(self.data_root, self.suffix, cleaned_fine_tree_fileName)
        self.tax = Taxonomy(node_ids, cleaned_fine_tree_fileName_full_path, verbose=False)

        # build distance matrix for efficiency
        fineList = self.getFineList()

        self.distance_matrix = torch.zeros(len(fineList), len(fineList))
        for i, species_i in enumerate(fineList):
            for j, species_j in enumerate(fineList):
                self.distance_matrix[i, j] = self.tax.get_distance(species_i, species_j)



    def get_target_from_layerName(self, batch, layer_name, hierarchyBased=True, z_triplet=None, triplet_layers_dic=['layer2', 'layer4']):
        first_layer = triplet_layers_dic[0]
        second_layer = None
        if hierarchyBased:
            second_layer = triplet_layers_dic[1]
        result = None
        
        if layer_name == first_layer:
            result = batch['coarse' if hierarchyBased==True else 'fine']
        elif (z_triplet is not None) and (second_layer in z_triplet):
            if (layer_name == second_layer):
                result = batch['fine']
        # elif (z_triplet is not None) and ('layer3' in z_triplet) and (layer_name == 'layer3'):
        #         result = batch['fine']
            
        return result


    def getComimicComponents(self):
        if self.comimics_components is None:
            fileFullPath = os.path.join(self.data_root, self.suffix, comimics_fileName)

            # get dataframe
            df = pd.read_csv(fileFullPath) 

            #clean up
            df = df[2:]
            df.columns = df.iloc[0]
            df = df[1:]

            # make readable
            df['FullSubspecies'] = df['Species'] + ' ' + df['Subspecies']
            df = df.drop(columns=['Species', 'Subspecies', 'Subspecies number'])
            df = df.set_index('FullSubspecies')

            # extract matrix of connectivity
            m = df.astype(int).to_numpy()
            self.comimics_components_lables = list(df.index.values)

            _, self.comimics_components = connected_components(csgraph=csr_matrix(m), directed=False, return_labels=True)

            # self.comimics_components = list(map(lambda x: df.index[x], self.comimics_components))

            print('comimics components:', self.comimics_components)


        return self.comimics_components, self.comimics_components_lables




# exmaple: FFFFffFF.JPG -> FFFFffFF_
def get_fileName_prefix(txt):
    return os.path.splitext(txt)[0]+"_"






# The following two functions should be changed together!!!


# Gets if csv_value is (exact or as prefix) in list 
# dir_lst is a tabbed string of concatenated values
# def filter(csv_value, dir_lst):
#     csv_in_dir = (str.upper(csv_value) in dir_lst)
#     if not csv_in_dir:
#         csv_prefix_in_dir = get_fileName_prefix(str.upper(csv_value)) in dir_lst
#         return csv_prefix_in_dir
#     return csv_in_dir

# Find an equivalent to the filename in the list by checking exact and prefix matches.
# This is used to get the name with same case as in the list (usually in directory)
def get_equivalent(csv_value, dir_lst):
    for i in dir_lst:
        if str.upper(csv_value)==str.upper(i):
            return i
        elif get_fileName_prefix(str.upper(csv_value)) in str.upper(i):
            return i
    return None
    





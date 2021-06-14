import time
import json
import os
import hashlib
import copy
import pandas as pd
import math


# Constants
configJsonFileName = "params.json"

######################
### helpers
def getDatasetParams(params):
    result = {
        "image_path": params["image_path"],
        "suffix": params['suffix'],
    }
    return result
    
def getModelName(params, trial_id=None):    
    modelName = str(params)
    if trial_id is not None:
        modelName = modelName + str(trial_id)  
    
    modelName = hashlib.sha224(modelName.encode('utf-8')).hexdigest()
    
    return os.path.join('models',modelName)

#########################
### Public methods

# The config parser is a helper object for creating experiments
class ConfigParser:
    def __init__(self, experimentsPath, dataPath, experimentName):
        self.experimentName = experimentName
        self.experimentsPath = experimentsPath
        self.experimentNameAndPath = os.path.join(self.experimentsPath, self.experimentName)
        self.dataPath = dataPath
        self.base_params = None
            
    def write(self, base_params, params):
        self.base_params = base_params
        
        fullFileName = os.path.join(self.experimentNameAndPath, configJsonFileName)
        if os.path.exists(self.experimentName) and os.path.exists(fullFileName):
            self.experimentName = self.experimentName+"-"+hex(int(time.time()))  
            self.experimentNameAndPath = os.path.join(self.experimentsPath, self.experimentName)
        fullFileName = os.path.join(self.experimentNameAndPath, configJsonFileName)
        
        if not os.path.exists(self.experimentNameAndPath):
            os.makedirs(self.experimentNameAndPath)

        # create experiment params
        experimentList = []
        for expriment in params:
            experimentList.append({**self.base_params, **expriment})
 
        # Write them
        j = json.dumps({"experimentList": experimentList})
        f = open(fullFileName,"w")        
        f.write(j)
        f.close()           

        return fullFileName

    def getExperiments(self, fixExperiments=True):
        fullFileName = os.path.join(self.experimentNameAndPath, configJsonFileName)
        if os.path.exists(fullFileName):
            with open(fullFileName, 'rb') as f:
                exp_list = json.loads(f.read())["experimentList"]
                experimentList = list(map(lambda x: self.fixExperimentParams(x) if fixExperiments else x , exp_list))

            return iter(experimentList)
        else:
            raise Exception('Error loading experiment parameters for ' + fullFileName) 
    
    

    def fixPaths(self, params_):
        params= copy.copy(params_)
        params["image_path"] = os.path.join(self.dataPath, params["image_path"])
        return params

    def fixExperimentParams(self, params_):
        params= copy.copy(params_)

        params["batchSize"] = params["batchSize"] if check_valid(params,"batchSize") else 32
        params["learning_rate"] = params["learning_rate"] if check_valid(params,"learning_rate") else 0.0005
        params["numOfTrials"] = params["numOfTrials"] if check_valid(params,"numOfTrials") else 1
        params["fc_layers"] = params["fc_layers"] if check_valid(params,"fc_layers") else 1
        params["modelType"] = params["modelType"] if check_valid(params,"modelType") else "BB"
        params["lambda"] = params["lambda"] if check_valid(params,"lambda") else 1
        params["tl_model"] = params["tl_model"] if check_valid(params,"tl_model") else "ResNet18"
        params["augmented"] = params["augmented"] if check_valid(params,"augmented") else False
        params["img_res"] = params["img_res"] if check_valid(params,"img_res") else 224
        params["link_layer"] = params["link_layer"] if check_valid(params,"link_layer") else "layer1"
        params["adaptive_smoothing"] = params["adaptive_smoothing"] if check_valid(params,"adaptive_smoothing") else False
        params["adaptive_lambda"] = params["adaptive_lambda"] if check_valid(params,"adaptive_lambda") else 0.1
        params["adaptive_alpha"] = params["adaptive_alpha"] if check_valid(params,"adaptive_alpha") else 0.9
        params["pretrained"] = params["pretrained"] if check_valid(params,"pretrained") else True

        return params

def check_valid(params, key):
     return (key in params) and (isinstance(params[key], str)or not math.isnan(params[key]))

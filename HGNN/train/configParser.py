import time
import json
import os
import itertools
import hashlib
import copy
import pandas as pd
import pickle
import math
    
#TODO: All experiments wit datasplit params having augmented should have it removed.
# This is because being augmented should have same split if it is not augmented.
def getDatasetParams(params):
    result = {
        "image_path": params["image_path"],
        "suffix": params['suffix'],
    }
    return result
    
def getDatasetName(params):
    datasetName = str(getDatasetParams(params))
    datasetName = hashlib.sha224(datasetName.encode('utf-8')).hexdigest()
    
    return os.path.join('datasplits',datasetName)
    
def getModelName(params, trial_id=None):    
    modelName = str(params)
    if trial_id is not None:
        modelName = modelName + str(trial_id)  
    
    modelName = hashlib.sha224(modelName.encode('utf-8')).hexdigest()
    
    return os.path.join('models',modelName)

experimetnsFileName = "experiments.csv"
paramsFileName="params.json"

def getExperimentParamsAndRecord(experimentsPath, experimentName, trial_hash) :
    experimentsFileNameAndPath = os.path.join(experimentsPath, experimetnsFileName)
    if os.path.exists(experimentsFileNameAndPath):
        experiments_df = pd.read_csv(experimentsFileNameAndPath)
        experimentRecord = experiments_df[experiments_df["trialHash"] == trial_hash]
        experimentRecord = experimentRecord[experimentRecord["experimentName"] == experimentName]
        # experiment_params = experimentRecord.to_dict('records')[0]

        modelName = experimentRecord.iloc[0]["modelName"]
        experimentNameAndPath = os.path.join(experimentsPath, experimentName)
        modelNameAndPath = os.path.join(experimentNameAndPath, modelName)
        fullFileName = os.path.join(modelNameAndPath, paramsFileName)
        with open(fullFileName, 'rb') as f:
            experiment_params = json.loads(f.read())
        return experiment_params, experimentRecord
    else:
        raise Exception("Experiment not " + trial_hash + " found!")





configJsonFileName = "params.json"
configPickleFileName = "params.pkl"

class ConfigParser:
    def __init__(self, experimentsPath, dataPath, experimentName):
        self.experimentName = experimentName
        self.experimentsPath = experimentsPath
        self.experimentNameAndPath = os.path.join(self.experimentsPath, self.experimentName)
        self.dataPath = dataPath
        self.base_params = None
            
    def write(self, base_params, params, experiment_type):
        self.base_params = base_params
        fileName = configJsonFileName if experiment_type != "Random" else configPickleFileName
        
        fullFileName = os.path.join(self.experimentNameAndPath, fileName)
        if os.path.exists(self.experimentName) and os.path.exists(fullFileName):
            self.experimentName = self.experimentName+"-"+hex(int(time.time()))  
            self.experimentNameAndPath = os.path.join(self.experimentsPath, self.experimentName)
        fullFileName = os.path.join(self.experimentNameAndPath, fileName)
        
        if not os.path.exists(self.experimentNameAndPath):
            os.makedirs(self.experimentNameAndPath)


        experimentList = []
        if experiment_type=="Grid":
            keys, values = zip(*params.items())
            # create experiment params
            for v in itertools.product(*values):
                experiment_params = dict(zip(keys, v))
                experimentList.append({**self.base_params, **experiment_params})
                
        elif experiment_type=="Random":
            for key in self.base_params:
                if key not in params:
                    params[key] = hp.choice(key, [self.base_params[key]])
                    
        elif experiment_type=="Select":
            # create experiment params
            for expriment in params:
                experimentList.append({**self.base_params, **expriment})
                
        else:
            raise Exception('Unknown experiment type') 
 
        
    
    
        if experiment_type=="Select" or experiment_type=="Grid":
            j = json.dumps({"experimentList": experimentList})
            f = open(fullFileName,"w")        
            f.write(j)
            f.close()           
        else:
            j = params
            with open(fullFileName, 'wb') as f:
                pickle.dump(params, f)
        
        return fullFileName

    def getExperiments(self, fixExperiments=True):
        fullFileName = os.path.join(self.experimentNameAndPath, configJsonFileName)
        if os.path.exists(fullFileName):
            with open(fullFileName, 'rb') as f:
                experimentList = list(map(lambda x: self.fixExperimentParams(x) if fixExperiments else x , json.loads(f.read())["experimentList"]))

            return iter(experimentList)
        else:
            raise Exception('Error loading experiment parameters for ' + fullFileName) 
    
    # For hyper param search, fixExperimentParams needs to be called outside. TODO: fix that requirement
    def getHyperoptSearchObject(self):
        fullFileName = os.path.join(self.experimentNameAndPath, configPickleFileName)
        if os.path.exists(fullFileName):   
            with open(fullFileName, 'rb') as f:
                hyperp_search_params = pickle.load(f)

            return hyperp_search_params
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
        params["noSpeciesBackprop"] = params["noSpeciesBackprop"] if check_valid(params,"noSpeciesBackprop") else False
        params["phylogeny_loss"] = params["phylogeny_loss"] if check_valid(params,"phylogeny_loss") else False
        params["phylogeny_loss_epsilon"] = params["phylogeny_loss_epsilon"] if check_valid(params,"phylogeny_loss_epsilon") else 0.03
        params["tripletEnabled"] = params["tripletEnabled"] if check_valid(params,"tripletEnabled") else False
        params["tripletSamples"] = params["tripletSamples"] if check_valid(params,"tripletSamples") else 10
        params["tripletSelector"] = params["tripletSelector"] if check_valid(params,"tripletSelector") else "semihard"
        params["tripletMargin"] = params["tripletMargin"] if check_valid(params,"tripletMargin") else 2

        return params

def check_valid(params, key):
     return (key in params) and (isinstance(params[key], str)or not math.isnan(params[key]))

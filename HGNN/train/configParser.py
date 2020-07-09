import time
import json
import os
import itertools
import hashlib
import copy
import pandas as pd
    
def getDatasetParams(params):
    return {
        "training_count": params["training_count"],
        "validation_count": params["validation_count"],
        "image_path": params["image_path"],
        "suffix": params['suffix'],
        "augmented": params['augmented'],
    }
    
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

        params["training_count"] = params["training_count"] if ("training_count" in params) is not None else 0.64
        params["validation_count"] = params["validation_count"] if ("validation_count" in params) is not None else 0.16
        params["batchSize"] = params["batchSize"] if ("batchSize" in params) else 32
        params["n_epochs"] = params["n_epochs"] if ("n_epochs" in params) else 10000
        params["patience"] = params["patience"] if ("patience" in params) else 100
        params["learning_rate"] = params["learning_rate"] if ("learning_rate" in params) else 0.0005
        params["fc_width"] = params["fc_width"] if ("fc_width" in params) else 200
        params["fc_layers"] = params["fc_layers"] if ("fc_layers" in params) else 1
        params["modelType"] = params["modelType"] if ("modelType" in params) else "blackbox"
        params["unsupervisedOnTest"] = params["unsupervisedOnTest"] if ("unsupervisedOnTest" in params) else False
        params["lambda"] = params["lambda"] if ("lambda" in params) else 1
        params["tl_model"] = params["tl_model"] if ("tl_model" in params) else "ResNet18"
        params["numOfTrials"] = params["numOfTrials"] if ("numOfTrials" in params) else 1
        params["tl_model"] = params["tl_model"] if ("tl_model" in params) else "ResNet18"
        params["augmented"] = params["augmented"] if ("augmented" in params) else False
        params["weight_decay"] = params["weight_decay"] if ("weight_decay" in params) else 0
        params["img_res"] = params["img_res"] if ("img_res" in params) else 224
        params["tl_freeze"] = params["tl_freeze"] if ("tl_freeze" in params) else True
        params["cnn_layers"] = params["cnn_layers"] if ("cnn_layers" in params) else 0
        params["cnn_channels"] = params["cnn_channels"] if ("cnn_channels" in params) else 32
        
        return params
import matplotlib.pyplot as plt
import torch
import sys
import os
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
from tqdm.auto import trange
import wandb

import warnings
# warnings.filterwarnings("ignore")

try:
    import wandb
except:
    print('wandb not found')

from myhelpers import config_plots, TrialStatistics
from myhelpers.try_warning import try_running
from HGNN.train import CNN, dataLoader
from HGNN.train.configParser import ConfigParser, getModelName, getDatasetName
config_plots.global_settings()

experimetnsFileName = "experiments.csv"
WANDB_message="wandb not working"

def main(experimentsPath, dataPath, experimentName, device=None, detailed_reporting=None):
    # For logging to server
    try_running(lambda : wandb.login(), WANDB_message)
    
    experimentPathAndName = os.path.join(experimentsPath, experimentName)
    # set cuda
    if device is not None:
        print("using cuda", device)
        torch.cuda.set_device(device)
    else:
        print("using cpu")

    # get experiment params
    config_parser = ConfigParser(experimentsPath, dataPath, experimentName)

    # init experiments file
    experimentsFileNameAndPath = os.path.join(experimentsPath, experimetnsFileName)
    
    # load data
    datasetManager = dataLoader.datasetManager(experimentPathAndName, dataPath)
    
    paramsIterator = config_parser.getExperiments()  
    number_of_experiments = sum(1 for e in paramsIterator)  
    experiment_index = 0

    # Loop through experiments
    # with progressbar.ProgressBar(max_value=number_of_experiments) as bar:
    with tqdm(total=number_of_experiments, desc="experiment") as bar:
        for experiment_params in config_parser.getExperiments():
            print(experiment_params)
            experimentHash =TrialStatistics.getTrialName(experiment_params)

            # load images 
            datasetManager.updateParams(config_parser.fixPaths(experiment_params))
            train_loader, validation_loader, test_loader = datasetManager.getLoaders()
            architecture = {
                "fine": len(train_loader.dataset.csv_processor.getFineList()),
                "coarse" : len(train_loader.dataset.csv_processor.getCoarseList())
            }

            # Loop through n trials
            for i in trange(experiment_params["numOfTrials"], desc="trial"):
                modelName = getModelName(experiment_params, i)
                trialName = os.path.join(experimentPathAndName, modelName)
                trialHash = TrialStatistics.getTrialName(experiment_params, i)

                row_information = {
                    'experimentName': experimentName,
                    'modelName': modelName,
                    'datasetName': getDatasetName(config_parser.fixPaths(experiment_params)),
                    'experimentHash': experimentHash,
                    'trialHash': trialHash
                }
                row_information = {**row_information, **experiment_params} 
                print(row_information)

                run = try_running(lambda : wandb.init(project='HGNN', group=experimentName+"-"+experimentHash, name=trialHash, config=row_information), WANDB_message) #, reinit=True

                # Train/Load model
                model = CNN.create_model(architecture, experiment_params, device=device)
                
                try_running(lambda : wandb.watch(model, log="all"), WANDB_message)
                
                if os.path.exists(CNN.getModelFile(trialName)):
                    print("Model {0} found!".format(trialName))
                else:
                    CNN.trainModel(train_loader, validation_loader, experiment_params, model, trialName, test_loader, device=device, detailed_reporting=detailed_reporting)

                # Add to experiments file
                if os.path.exists(experimentsFileNameAndPath):
                    experiments_df = pd.read_csv(experimentsFileNameAndPath)
                else:
                    experiments_df = pd.DataFrame()

                record_exists = not (experiments_df[experiments_df['modelName'] == modelName][experiments_df['experimentName'] == experimentName]).empty if not experiments_df.empty else False
                if record_exists:
                    experiments_df.drop(experiments_df[experiments_df['modelName'] == modelName][experiments_df['experimentName'] == experimentName].index, inplace = True) 

                experiments_df = experiments_df.append(pd.DataFrame(row_information, index=[0]), ignore_index = True)
                experiments_df.to_csv(experimentsFileNameAndPath, header=True, index=False)


                try_running(lambda : run.finish(), WANDB_message)

            bar.update()
            
            experiment_index = experiment_index + 1
        
            

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', required=True, type=int)
    parser.add_argument('--experiments', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--detailed', required=False, action='store_true')
    args = parser.parse_args()
    main(experimentName=args.name, experimentsPath=args.experiments, dataPath=args.data, device=args.cuda, detailed_reporting=args.detailed)
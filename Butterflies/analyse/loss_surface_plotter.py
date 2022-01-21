import os
from HGNN.train import CNN
import torch
from tqdm.auto import tqdm 

import loss_landscapes

final_model_name = "finalModel.pt"
iterations_folder_name = "iterations"

# get all optimization path models
def fetch_model_paths(root_path):
    model_paths = []

    i=0
    while(True):
        p = os.path.join(root_path, iterations_folder_name, "iteration{0}.pt".format(i))
        if os.path.exists(p):
            model_paths.append(p)
            i = i+1
        else:
            model_paths.append(os.path.join(root_path, iterations_folder_name, final_model_name))
            print(i+1, " models added")
            break
    return model_paths


def get_models(model_paths, architecture, experiment_params):
    models = []
    for model_path in tqdm(model_paths):
        model = CNN.create_model(architecture, experiment_params)
        model.load_state_dict(torch.load(os.path.join(model_path))) # , map_location=torch.device('cpu')
        model.eval()
        models.append(model)
    return models
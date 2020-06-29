from torch import nn
import os
import torch
import csv
import time
from scipy.stats import entropy
import collections
import pandas as pd
import torchvision.models as models
from torch.nn import Module
from sklearn.metrics import f1_score
import json
from tqdm import tqdm

from myhelpers.earlystopping import EarlyStopping
from .resnet_cifar import cifar_resnet56
from .cifar_nin import nin_cifar100




import time


modelFinalCheckpoint = 'finalModel.pt'

statsFileName = "stats.csv"
timeFileName = "time.csv"
epochsFileName = "epochs.csv"

paramsFileName="params.json"


class ZeroModule(Module):
    def __init__(self, *args, **kwargs):
        super(torch.nn.Identity, self).__init__()

    def forward(self, input):
        return torch.zeros_like(input)

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# Create an FC layer with RELU and/or BatchNormalization
def get_fc(num_of_inputs, num_of_outputs, num_of_layers = 1):
    l = [] 
    
    for i in range(num_of_layers):
        n_out = num_of_inputs if (i+1 != num_of_layers) else num_of_outputs
        l.append(('linear'+str(i), torch.nn.Linear(num_of_inputs, n_out)))
        l.append(('bnorm'+str(i), torch.nn.BatchNorm1d(n_out)))
        l.append(('relu'+str(i), torch.nn.ReLU()))
        
    d = collections.OrderedDict(l)
    seq = torch.nn.Sequential(d)
    
    return seq

def create_pretrained_model(params):
    tl_model = params["tl_model"]
    
    if tl_model == "NIN":
        model = nin_cifar100(pretrained=True)
    elif tl_model == "CIFAR":
        model = cifar_resnet56(pretrained='cifar100')
    elif tl_model == "ResNet18":
        model = models.resnet18(pretrained=True)
    elif tl_model == "ResNet50":
        model = models.resnet50(pretrained=True)
    else:
        raise Exception('Unknown network type')
        
    for param in model.parameters():
        param.requires_grad = False
        
    if tl_model != "NIN":
        num_ftrs = model.fc.in_features
    else:
        num_ftrs =100
        
    return model, num_ftrs

def create_model(architecture, params):
    model = None

    if params["modelType"] != "basic_blackbox":
        model = CNN_hierarchy(architecture, params)
    else:  
        tl_model = params["tl_model"]
        fc_layers = params["fc_layers"]
        
        model, num_ftrs = create_pretrained_model(params)
        fc = get_fc(num_ftrs, architecture["fine"], fc_layers)
        if tl_model != "NIN":
            model.fc = fc
        else:
            model.output.add_module("fc", fc)
            

    if torch.cuda.is_available():
        model = model.cuda()
    return model


def getCustomTL_layer(tl_model, pretrained_model):
    if tl_model == "NIN":
        output = torch.nn.Sequential()
        output.add_module("pretrained", pretrained_model)
        output.add_module("fc", Flatten())
        return output
    else:
        if tl_model == "CIFAR":
            tl_model_subLayers = [pretrained_model.conv1,
              pretrained_model.bn1,
              pretrained_model.relu,
              pretrained_model.layer1,
              pretrained_model.layer2,
              pretrained_model.layer3,
              pretrained_model.avgpool]
        else:
            tl_model_subLayers = [pretrained_model.conv1,
              pretrained_model.bn1,
              pretrained_model.relu,
              pretrained_model.maxpool,
              pretrained_model.layer1,
              pretrained_model.layer2,
              pretrained_model.layer3,
              pretrained_model.layer4,
              pretrained_model.avgpool]
        return torch.nn.Sequential(*tl_model_subLayers, Flatten())

# Build a Hierarchical convolutional Neural Network
class CNN_hierarchy(nn.Module):
    
    # Contructor
    def __init__(self, architecture, params):
        modelType = params["modelType"]
        self.numberOfFine = architecture["fine"]
        self.numberOfCoarse = architecture["coarse"] if not modelType=="DSN" else architecture["fine"]
        fc_width = params["fc_width"]
        fc_layers = params["fc_layers"]
        tl_model = params["tl_model"]
        
        super(CNN_hierarchy, self).__init__()

        # The pretrained model
        self.pretrained_model, num_ftrs = create_pretrained_model(params)
        self.custom_tl_layer = getCustomTL_layer(tl_model, self.pretrained_model)
        
        # g_c block
        self.g_c = None
        g_c_num_ftrs = fc_width
        if modelType == "HGNNgcI":
            self.g_c = torch.nn.Identity()
        elif modelType != "HGNNgc0" or modelType != "BB":
            self.g_c = get_fc(fc_width, self.numberOfCoarse, num_of_layers=fc_layers)
            g_c_num_ftrs = self.numberOfCoarse
        
        # h_y block
        self.h_y = get_fc(num_ftrs, fc_width, num_of_layers=fc_layers)            
            
        # h_b block
        h_b_num_ftrs = 0
        self.h_b = None
        if modelType == "HGNNhbI":
            self.h_b = torch.nn.Identity()
            h_b_num_ftrs = num_ftrs
        elif modelType != "DISCO" and modelType != "DSN" and modelType != "BB" :
            self.h_b = get_fc(num_ftrs, fc_width, num_of_layers=fc_layers)
            h_b_num_ftrs = fc_width

            
        # g_y block
        self.g_y = get_fc(fc_width + h_b_num_ftrs, self.numberOfFine, num_of_layers=fc_layers)

        if torch.cuda.is_available():
            self.custom_tl_layer = self.custom_tl_layer.cuda()
            self.g_y = self.g_y.cuda()
            self.h_y = self.h_y.cuda()
            if self.g_c is not None:
                self.g_c = self.g_c.cuda()
            if self.h_b is not None:
                self.h_b = self.h_b.cuda()
    
    # Prediction
    def forward(self, x):
        activations = self.activations(x)
        result = {
            "fine": activations["fine"],
            "coarse" : activations["coarse"]
        }
        return result


    default_outputs = {
        "fine": True,
        "coarse" : True
    }
    def activations(self, x, outputs=default_outputs):
        tl_features = self.custom_tl_layer(x)
        
        hy_features = self.h_y(tl_features)
        
        hb_hy_features = None
        hb_features = None
        if self.h_b is not None:
            hb_features = self.h_b(tl_features)
            hb_hy_features = torch.cat((hy_features, hb_features), 1)
        else:
            hb_hy_features = hy_features
            
        yc = None
        if outputs["coarse"] and self.g_c is not None:
            yc = self.g_c(hy_features)
        
        species = None
        if outputs["fine"]:
            y = self.g_y(hb_hy_features)
            

        activations = {
            "input": x,
            "tl_features": tl_features,
            "hy_features": hy_features,
            "hb_features": hb_features,
            "coarse": yc if outputs["coarse"] else None,
            "fine": y if outputs["fine"] else None
        }

        return activations

def getModelFile(experimentName):
    return os.path.join(experimentName, modelFinalCheckpoint)


def trainModel(train_loader, validation_loader, params, model, savedModelName, test_loader=None):  
    n_epochs = params["n_epochs"]
    patience = params["patience"]
    learning_rate = params["learning_rate"]
    modelType = params["modelType"]
    batchSize = params["batchSize"]
    unsupervisedOnTest = params["unsupervisedOnTest"]
    lambda_ = params["lambda"]
    isOldBlackbox = (modelType == "basic_blackbox")
    isDSN = (modelType == "DSN")
    
    df = pd.DataFrame()
    
    if not os.path.exists(savedModelName):
        os.makedirs(savedModelName)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    # early stopping
    early_stopping = EarlyStopping(path=savedModelName, patience=patience)

    print("Training started...")
    start = time.time()
    with tqdm(total=n_epochs, desc="iteration") as bar:
        epochs = 0
        for epoch in range(n_epochs):
            criterion = nn.CrossEntropyLoss()
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    z = applyModel(batch["image"], model)
                    
                    if not isOldBlackbox:
                        loss_coarse = 0
                        if z["coarse"] is not None:
                            loss_coarse = criterion(z["coarse"], batch["coarse"] if not isDSN else batch["fine"])
                        loss_fine = criterion(z["fine"], batch["fine"])
                        loss = loss_fine + lambda_*loss_coarse
                        loss.backward()
                    else:    
                        loss_fine = criterion(z, batch["fine"])
                        loss_fine.backward()
                    optimizer.step()
            if unsupervisedOnTest and test_loader and not isOldBlackbox:
                for batch in test_loader:
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        z = applyModel(batch["image"], model)

                        loss_unsupervised = criterion(z["coarse"], batch["coarse"])
                        loss_unsupervised.backward()
                        optimizer.step()
            
            model.eval()
            
            row_information = {
                'validation_fine_f1': getLoader_f1(validation_loader, model, params),
                'training_fine_f1': getLoader_f1(train_loader, model, params),
                'test_fine_f1': getLoader_f1(test_loader, model, params) if test_loader else None,
                'validation_loss': getCrossEntropyFromLoader(validation_loader, model, params),
                'training_loss': getCrossEntropyFromLoader(train_loader, model, params),

                'training_coarse_loss': getCrossEntropyFromLoader(train_loader, model, params, "coarse") if not isOldBlackbox and not isDSN else None,
                'validation_coarse_loss': getCrossEntropyFromLoader(validation_loader, model, params, "coarse") if not isOldBlackbox and not isDSN else None,
                'training_coarse_f1': getLoader_f1(train_loader, model, params, "coarse") if not isDSN else None,
                'validation_coarse_f1': getLoader_f1(validation_loader, model, params, "coarse")if not isDSN else None,
                'test_coarse_f1': getLoader_f1(test_loader, model, params, "coarse") if test_loader and not isDSN else None,
            }
            
            df = df.append(pd.DataFrame(row_information, index=[0]), ignore_index = True)
            
            # Update the bar
            bar.set_postfix(val=row_information["validation_fine_f1"], 
                            train=row_information["training_fine_f1"],
                            loss=row_information["training_loss"],)
            bar.update()

            # early stopping
            early_stopping(row_information['validation_loss'], epoch, model)

            epochs = epochs + 1
            if early_stopping.early_stop:
                print("Early stopping")
                print("total number of epochs: ", epoch)
                break
            
        
        # Register time
        end = time.time()
        time_elapsed = end - start
        
        # load the last checkpoint with the best model
        model.load_state_dict(early_stopping.getBestModel())
        
        # save information
        if savedModelName is not None:
            # save model
            torch.save(model.state_dict(), os.path.join(savedModelName, modelFinalCheckpoint))
            # save results
            df.to_csv(os.path.join(savedModelName, statsFileName))  
            
            with open(os.path.join(savedModelName, timeFileName), 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerow([time_elapsed])
            with open(os.path.join(savedModelName, epochsFileName), 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerow([epochs])
            # save params
            j = json.dumps(params)
            f = open(os.path.join(savedModelName, paramsFileName),"w")        
            f.write(j)
            f.close()  
    
    return df, epochs, time_elapsed

# loads a saved model along with its results
def loadModel(model, savedModelName):
    model.load_state_dict(torch.load(os.path.join(savedModelName, modelFinalCheckpoint))) 
    model.eval()

    time_elapsed = 0
    epochs = 0
    
    df = pd.read_csv(os.path.join(savedModelName, statsFileName))
    
    with open(os.path.join(savedModelName, timeFileName), newline='') as f:
        reader = csv.reader(f)
        time_elapsed = float(next(reader)[0])
    with open(os.path.join(savedModelName, epochsFileName), newline='') as f:
        reader = csv.reader(f)
        epochs = float(next(reader)[0])
        
    return df, epochs, time_elapsed

def top_k_acc(output, target, topk=(1,2,3,4,5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Returns the mean of CORRECT probability of all predictions. If high, it means the model is sure about its predictions
def getAvgProbCorrectGuessFromLoader(loader, model, params, label="fine"):
    predlist, lbllist = getLoaderPredictionProbabilities(loader, model, params, label)
    lbllist = lbllist.reshape(lbllist.shape[0], -1)
    predlist = predlist.gather(1, lbllist)
    max_predlist = predlist.mean().item()
    return max_predlist

# # Returns the mean of best probability of all predictions. If low, it means the model is sure about its predictions
# def getAvgEntropyFromLoader(loader, model, params, label="fine"):
#     predlist, _ = getLoaderPredictionProbabilities(loader, model, params, label)
#     return torch.Tensor(entropy(predlist.cpu().T, base=2)).mean().item()

def getCrossEntropyFromLoader(loader, model, params, label="fine"):
    predlist, lbllist = getLoaderPredictionProbabilities(loader, model, params, label) 

    criterion = nn.CrossEntropyLoss()
    return criterion(predlist, lbllist).item()

def getLoaderPredictionProbabilities(loader, model, params, label="fine"):
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0)
    lbllist=torch.zeros(0, dtype=torch.long)
    isOldBlackbox = (params['modelType'] == "basic_blackbox")
    
    if torch.cuda.is_available():
        predlist = predlist.cuda()
        lbllist = lbllist.cuda()

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in loader:
            inputs = batch["image"]
            classes = batch[label]
            preds = applyModel(inputs, model)
            if not isOldBlackbox:
                preds = preds[label]
            preds = torch.nn.Softmax(dim=1)(preds)

            # Append batch prediction results
            predlist=torch.cat([predlist,preds], 0)
            lbllist=torch.cat([lbllist,classes], 0)  
            
    return predlist, lbllist



def getLoaderPredictions(loader, model, params, label="fine"):
    predlist, lbllist = getLoaderPredictionProbabilities(loader, model, params, label)
    _, predlist = torch.max(predlist, 1)
    
    if torch.cuda.is_available():
        predlist = predlist.cpu()
        lbllist = lbllist.cpu()     
        
    return predlist, lbllist

def getLoader_f1(loader, model, params, label="fine"):
    predlist, lbllist = getLoaderPredictions(loader, model, params, label)
    return f1_score(lbllist, predlist, average='macro')


def applyModel(batch, model):
#     if torch.cuda.is_available():
#         model_dist = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
#         outputs = model_dist(batch)
#     else:
    outputs = model(batch)
    return outputs
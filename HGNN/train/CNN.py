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
import math

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

# Create an FC layer with RELU and BatchNormalization
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

def get_output_res(input_res, kernel_size, stride, padding):
    return math.floor((input_res + 2*padding - kernel_size)/stride +1)

# Create an Conv layer with RELU and BatchNormalization
def get_conv(input_res, output_res, input_num_of_channels, intermediate_num_of_channels, output_num_of_channels, num_of_layers = 1, kernel_size=3, stride=1, padding=1):
    #  Sanity checks 
    assert(input_res >= output_res)
    needed_downsampling_layers=0
    res = input_res
    for i in range(num_of_layers):
        intermediate_output_res = get_output_res(res, kernel_size, stride, padding)
        assert(intermediate_output_res <= res)
        needed_downsampling_layers = needed_downsampling_layers + 1
        res = intermediate_output_res
        if intermediate_output_res == output_res:
            break

    l = [] 
    
    # First k layers no downsampling
    in_ = input_num_of_channels
    for i in range(num_of_layers - needed_downsampling_layers):
        out_ = intermediate_num_of_channels if i<num_of_layers-1 else output_num_of_channels
        l.append(('conv'+str(i), torch.nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)))
        l.append(('bnorm'+str(i), torch.nn.BatchNorm2d(out_)))
        l.append(('relu'+str(i), torch.nn.ReLU()))
        in_ = out_

    # Then downsample each remaining layer till we get to the desired output resolution 
    for i in range(needed_downsampling_layers):
        out_ = output_num_of_channels if i + num_of_layers - needed_downsampling_layers == num_of_layers-1 else intermediate_num_of_channels
        l.append(('conv'+str(i+needed_downsampling_layers), torch.nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)))
        l.append(('bnorm'+str(i+needed_downsampling_layers), torch.nn.BatchNorm2d(out_)))
        l.append(('relu'+str(i+needed_downsampling_layers), torch.nn.ReLU()))
        in_ = out_

    d = collections.OrderedDict(l)
    seq = torch.nn.Sequential(d)
    
    return seq

def create_pretrained_model(params):
    tl_model = params["tl_model"]
    tl_freeze = False
    
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
        
    if tl_model != "NIN":
        num_ftrs = model.fc.in_features
    else:
        num_ftrs =100
        
    return model, num_ftrs

def create_model(architecture, params):
    model = None

    if params["modelType"] != "basic_blackbox":
        model = CNN_Two_Nets(architecture, params)
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


# Gets a customized transfer learning model between two layers.
# from_layer=None => from first layer. Otherwise, pass layer name
# to_layer=None => to last layer. Otherwise, pass layer name
# Returns [from_layer, to_layer). i.e. t_layer not included.
def getCustomTL_layer(tl_model, pretrained_model, from_layer=None, to_layer=None):
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
            # tl_model_subLayers = [pretrained_model.conv1,
            #   pretrained_model.bn1,
            #   pretrained_model.relu,
            #   pretrained_model.maxpool,
            #   pretrained_model.layer1,
            #   pretrained_model.layer2,
            #   pretrained_model.layer3,
            #   pretrained_model.layer4,
            #   pretrained_model.avgpool]

            # Get layers except for fc (last) one
            tl_model_subLayer_names = list(dict(pretrained_model.named_children()).keys())[:-1]

            # Get indices of from and to layers
            from_layer_index = 0
            to_layer_index = len(tl_model_subLayer_names)
            if from_layer is not None:
                try:
                    from_layer_index = tl_model_subLayer_names.index(from_layer)
                except:
                    print(from_layer, "is not in", tl_model_subLayer_names)
                    raise
            if to_layer is not None:
                try:
                    to_layer_index = tl_model_subLayer_names.index(to_layer)
                except:
                    print(to_layer, "is not in", tl_model_subLayer_names)
                    raise

            children_layers = list(pretrained_model.children())
            tl_model_subLayer_names_subset = tl_model_subLayer_names[from_layer_index:to_layer_index]
            tl_model_subLayers = list(map(lambda x: children_layers[tl_model_subLayer_names.index(x)], tl_model_subLayer_names_subset))
        return tl_model_subLayers

def get_layer_by_name(model, layer_name):
    model_subLayer_names = list(dict(model.named_children()).keys())[:-1]
    children_layers = list(model.children())
    return children_layers[model_subLayer_names.index(layer_name)]

# Build a Hierarchical convolutional Neural Network with conv layers
class CNN_Two_Nets(nn.Module):  
    # Contructor
    def __init__(self, architecture, params):
        modelType = params["modelType"]
        self.modelType = modelType
        self.numberOfFine = architecture["fine"]
        self.numberOfCoarse = architecture["coarse"] if not modelType=="DSN" else architecture["fine"]
        tl_model = params["tl_model"]
        link_layer = params["link_layer"]
        fc_layers = params["fc_layers"]
        img_res = params["img_res"]
        
        super(CNN_Two_Nets, self).__init__()

        # The pretrained models
        self.network_coarse, num_ftrs_coarse = create_pretrained_model(params)
        self.network_fine, num_ftrs_fine = create_pretrained_model(params)
        
        # h_y block
        self.h_y = torch.nn.Sequential(*getCustomTL_layer(tl_model, self.network_coarse, None, link_layer)) 

        # g_c block
        self.g_c = None
        if modelType == "HGNNgcI":
            # TODO: fix this case. broken because we need avg pooling from h_y
            print("HGNNgcI case not supported yet")
            raise 
            self.g_c = [torch.nn.Identity()]
        elif modelType != "HGNNgc0" and modelType != "BB":
            self.g_c = getCustomTL_layer(tl_model, self.network_coarse, link_layer, None)
        if self.g_c is not None:
            self.g_c = torch.nn.Sequential(*self.g_c, Flatten())
            self.g_c_fc = get_fc(num_ftrs_coarse, self.numberOfCoarse, num_of_layers=fc_layers)
                    
        # h_b block
        self.h_b = None
        if modelType == "HGNNhbI":
            # TODO: To fix this case, we need img and hy to have same dimensions 
            print("HGNNgcI case not supported yet")
            raise 
            self.h_b = torch.nn.Identity()
        elif modelType != "DISCO" and modelType != "DSN" and modelType != "BB" :
            self.h_b = torch.nn.Sequential(*getCustomTL_layer(tl_model, self.network_fine, None, link_layer))

        # h_b + h_y -> g_y
        self.cat_conv2d = None
        if self.h_b is not None:
            if modelType == "HGNN":
                # concatenate hb and hy features and then cut the number of channels by 2
                hb_features = self.h_b(torch.rand(1, 3, img_res, img_res))
                hy_features = self.h_y(torch.rand(1, 3, img_res, img_res))
                assert(hy_features.shape == hb_features.shape), "hb and hy activations should be of same size" 
                assert(hb_features.shape[2] == hb_features.shape[3]), "hb/hy should be square-shaped"
                hb_hy_features = torch.cat((hy_features, hb_features), 1)
                resolution = hb_features.shape[2]
                in_channels = hb_hy_features.shape[1]
                self.cat_conv2d = get_conv(resolution, resolution, in_channels, in_channels, int(in_channels/2))
                if torch.cuda.is_available():
                    self.cat_conv2d = self.cat_conv2d.cuda()

        # g_y block
        self.g_y = torch.nn.Sequential(*getCustomTL_layer(tl_model, self.network_fine, link_layer, None),  
                                       Flatten())
        self.g_y_fc = get_fc(num_ftrs_fine, self.numberOfFine, num_of_layers=fc_layers)

        if torch.cuda.is_available():
            self.g_y = self.g_y.cuda()
            self.h_y = self.h_y.cuda()
            self.g_y_fc = self.g_y_fc.cuda()
            if self.g_c is not None:
                self.g_c = self.g_c.cuda()
                self.g_c_fc = self.g_c_fc.cuda()
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
        hy_features = self.h_y(x)
        
        hb_hy_features = None
        hb_features = None
        if self.h_b is not None:
            hb_features = self.h_b(x)
            if self.modelType == "HGNN":
                hb_hy_features = torch.cat((hy_features, hb_features), 1)
                hb_hy_features =  self.cat_conv2d(hb_hy_features)
            elif self.modelType == "HGNN_cat":
                hb_hy_features = torch.cat((hy_features, hb_features), 2)
            else:
                hb_hy_features = hy_features + hb_features

        else:
            hb_hy_features = hy_features

        yc = None
        gc_features = None
        if outputs["coarse"] and self.g_c is not None:
            gc_features = self.g_c(hy_features)
            yc = self.g_c_fc(gc_features)
        
        y = None
        gy_features = None
        if outputs["fine"]:
            gy_features = self.g_y(hb_hy_features)
            y = self.g_y_fc(gy_features)
            

        modelType_has_coarse = gc_features is not None and (self.modelType!="DSN")  

        activations = {
            "input": x,
            "hy_features": hy_features,
            "hb_features": hb_features,
            "gy_features": gy_features if outputs["fine"] else None,
            "gc_features": gc_features if outputs["coarse"] else None,
            "coarse": yc if outputs["coarse"] and modelType_has_coarse else None,
            "fine": y if outputs["fine"] else None
        }

        return activations


def getModelFile(experimentName):
    return os.path.join(experimentName, modelFinalCheckpoint)


def trainModel(train_loader, validation_loader, params, model, savedModelName, test_loader=None):  
    n_epochs = 500
    patience = 5
    learning_rate = params["learning_rate"]
    modelType = params["modelType"]
    batchSize = params["batchSize"]
    unsupervisedOnTest = params["unsupervisedOnTest"]
    lambda_ = params["lambda"]
    weight_decay = 0.0001
    isOldBlackbox = (modelType == "basic_blackbox")
    isBlackbox = (modelType == "BB")
    isDSN = (modelType == "DSN")
    
    df = pd.DataFrame()
    
    if not os.path.exists(savedModelName):
        os.makedirs(savedModelName)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    
    # early stopping
    early_stopping = EarlyStopping(path=savedModelName, patience=patience)

    print("Training started...")
    start = time.time()
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    with tqdm(total=n_epochs, desc="iteration") as bar:
        epochs = 0
        for epoch in range(n_epochs):
            model.train()
            for batch in train_loader:
    
                if torch.cuda.is_available():
                    batch["image"] = batch["image"].cuda()
                    batch["fine"] = batch["fine"].cuda()
                    batch["coarse"] = batch["coarse"].cuda()

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

                    if torch.cuda.is_available():
                        batch["image"] = batch["image"].cuda()
                        batch["fine"] = batch["fine"].cuda()
                        batch["coarse"] = batch["coarse"].cuda()

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

                'training_coarse_loss': getCrossEntropyFromLoader(train_loader, model, params, "coarse") if not isOldBlackbox and not isDSN and not isBlackbox else None,
                'validation_coarse_loss': getCrossEntropyFromLoader(validation_loader, model, params, "coarse") if not isOldBlackbox and not isDSN and not isBlackbox else None,
                'training_coarse_f1': getLoader_f1(train_loader, model, params, "coarse") if not isDSN else None,
                'validation_coarse_f1': getLoader_f1(validation_loader, model, params, "coarse")if not isDSN else None,
                'test_coarse_f1': getLoader_f1(test_loader, model, params, "coarse") if test_loader and not isDSN else None,
            }
            
            df = df.append(pd.DataFrame(row_information, index=[0]), ignore_index = True)
            
            # Update the bar
            bar.set_postfix(val=row_information["validation_fine_f1"], 
                            train=row_information["training_fine_f1"],
                            val_loss=row_information["validation_loss"],
                            min_val_loss=early_stopping.val_loss_min)
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
    model.load_state_dict(torch.load(os.path.join(savedModelName, modelFinalCheckpoint), map_location=torch.device('cpu'))) 
    if torch.cuda.is_available():
        print('model',model.get_device())
        model.cuda()
        print('model',model.get_device())
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
            if torch.cuda.is_available():
                batch["image"] = batch["image"].cuda()
                batch["fine"] = batch["fine"].cuda()
                batch["coarse"] = batch["coarse"].cuda()

            inputs = batch["image"]
            classes = batch[label]
            preds = applyModel(inputs, model)
            if not isOldBlackbox:
                if label in preds and preds[label] is not None:
                    preds = preds[label]
                elif label == 'coarse':
                    fineToCoarseMatrix = loader.dataset.csv_processor.getFineToCoarseMatrix()
                    if torch.cuda.is_available():
                        fineToCoarseMatrix = fineToCoarseMatrix.cuda()
                    preds = torch.mm(preds['fine'],fineToCoarseMatrix) 
                else:
                    raise
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

# Returns the distance between examples in terms of classification cross entropy
# Augmentation should be disabled
def get_distance_from_example(dataset, example_1, model, params):
    isOldBlackbox = (params['modelType'] == "basic_blackbox")
    criterion = torch.nn.CosineSimilarity()
    dataset_size = len(dataset)
    result = torch.zeros(1, dataset_size)

    with torch.set_grad_enabled(True):
        z_1 = applyModel(example_1["image"].unsqueeze(0), model)
        if not isOldBlackbox:
            z_1 = z_1['fine']
        z_1 = z_1.detach()
        z_1 = torch.nn.Softmax(dim=1)(z_1)
        for j in range(dataset_size):
            example_2 = dataset[j]
            z_2 = applyModel(example_2["image"].unsqueeze(0), model)

            if not isOldBlackbox:
                z_2 = z_2['fine']

            z_2 = z_2.detach()
            z_2 = torch.nn.Softmax(dim=1)(z_2)

            loss = criterion(z_1, z_2)


            result[0, j] = loss
    return result


def applyModel(batch, model):
#     if torch.cuda.is_available():
#         model_dist = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
#         outputs = model_dist(batch)
#     else:
    outputs = model(batch)
    return outputs
from torch import nn
import torch
import progressbar
from earlystopping import EarlyStopping
import os
from torch.autograd import Variable
import csv
import time
from sklearn.metrics import confusion_matrix
import collections

CheckpointNameFinal = 'finalModel.pt'

accuracyFileName = "validation_accuracy.csv"
lossFileName = "training_loss.csv"
timeFileName = "time.csv"
epochsFileName = "epochs.csv"

import torchvision.models as models

# Create an FC layer with RELU and/or BatchNormalization
def get_fc(num_of_inputs, num_of_outputs, with_relu = False, with_bnorm = False, num_of_layers = 1):
    l = [] 
    
    for i in range(num_of_layers):
        n_out = num_of_inputs if (i+1 != num_of_layers) else num_of_outputs
        l.append(('linear'+str(i), torch.nn.Linear(num_of_inputs, n_out)))
        if with_bnorm:
            l.append(('bnorm'+str(i), torch.nn.BatchNorm1d(n_out)))
        if with_relu:
            l.append(('relu'+str(i), torch.nn.ReLU()))
    d = collections.OrderedDict(l)
    
    seq = torch.nn.Sequential(d)
    return seq

def create_pretrained_model():
    
    model = models.resnet18(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    return model, num_ftrs

def create_model():
    model = None

    print('using a pretrained resnet model...')
    model, num_ftrs = create_pretrained_model()
    fc = get_fc(num_ftrs, 2, True, True, 2)
    fc = torch.nn.Sequential(fc, torch.nn.Softmax(dim=1))
    model.fc = fc
    
    from torchsummary import summary
    summary(model, (3, 224, 224))

    return model

def getModelFile(experimentName):
    return os.path.join(experimentName, CheckpointNameFinal)
    
def trainModel(train_loader, validation_loader, model, savedModelName):
    n_epochs = 10000
    patience = 100
    learning_rate = 0.01
    
    if not os.path.exists(savedModelName):
        os.makedirs(savedModelName)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    training_loss_list=[]
    validation_accuracy_list=[]
    
    # early stopping
    early_stopping = EarlyStopping(path=savedModelName, patience=patience)

    print("Training started...")
    start = time.time()
    with progressbar.ProgressBar(maxval=n_epochs, redirect_stdout=True) as bar:
        bar.update(0)
        epochs = 0
        for epoch in range(n_epochs):
            criterion = nn.CrossEntropyLoss()
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    z = applyModel(batch["image"], model)
                    loss = None
                    loss = criterion(z, batch["class"])
                    loss.backward()
                    optimizer.step()
            model.eval()

            #perform a prediction on the validation data  
            validation_accuracy_list.append(getAccuracyFromLoader(validation_loader, model))
            training_loss_list.append(loss.data.item())
            
            bar.update(epoch+1)
            
            # early stopping
            early_stopping(loss.data, epoch, model)

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
            torch.save(model.state_dict(), os.path.join(savedModelName, CheckpointNameFinal))
    
    return training_loss_list, validation_accuracy_list, epochs, time_elapsed

# loads a saved model along with its results
def loadModel(model, savedModelName):
    model.load_state_dict(torch.load(os.path.join(savedModelName, CheckpointNameFinal))) 
    model.eval()

def getAccuracyFromLoader(loader, model):
    correct=0
    N_test=0
    model.eval()
    for batch in loader:
        with torch.set_grad_enabled(False):
            z = applyModel(batch["image"], model)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == batch["class"]).sum().item()
            N_test = N_test + len(batch["image"])
    return correct / N_test

def getCrossEntropyFromLoader(loader, model):
    predlist, lbllist = getLoaderPredictions(loader, model, False) 

    criterion = nn.CrossEntropyLoss()
    return criterion(predlist, lbllist).item()

def getLoaderPredictionProbabilities(loader, model):
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0)
    lbllist=torch.zeros(0, dtype=torch.long)

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in loader:
            inputs = batch["image"]
            classes = batch["class"].unsqueeze(1)
            preds = applyModel(inputs, model)

            # Append batch prediction results
            predlist=torch.cat([predlist,preds], 0)
            lbllist=torch.cat([lbllist,classes], 0)  
            
    return predlist, lbllist

def getLoaderPredictions(loader, model, flattenConcat = True):
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0)
    if flattenConcat:
        lbllist=torch.zeros(0)
    else:
        lbllist=torch.zeros(0, dtype=torch.long)

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in loader:
            inputs = batch["image"]
            classes = batch["class"]
            outputs = applyModel(inputs, model)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            if flattenConcat:
                predlist=torch.cat([predlist,preds.float().view(-1)])
                lbllist=torch.cat([lbllist,classes.float().view(-1)])
            else:
                predlist=torch.cat([predlist,outputs], 0)
                lbllist=torch.cat([lbllist,classes], 0)  
            
    return predlist, lbllist


def applyModel(batch, model):
    outputs = model(batch)
    return outputs
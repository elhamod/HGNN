from torch import nn
import os
import torch
import csv
import time
# from scipy.stats import entropy
import collections
import pandas as pd
import torchvision.models as models
from torch.nn import Module
from sklearn.metrics import f1_score, accuracy_score
import json
from tqdm.notebook import tqdm
from torchsummary import summary
from adabelief_pytorch import AdaBelief
import random
from sklearn.preprocessing import MultiLabelBinarizer


from .criterion_phylogeny_KLDiv import Phylogeny_KLDiv
from .criterion_phylogeny_MSE import Phylogeny_MSE

try:
    import wandb
except:
    print('wandb not found')

from myhelpers.earlystopping import EarlyStopping
from myhelpers.try_warning import try_running
from myhelpers.resnet_cifar import cifar_resnet56
from myhelpers.resnet_cifar2 import cifar100
from myhelpers.preresnet_cifar import resnet as preresnet_cifar
from myhelpers.resnet import resnet56
from myhelpers.adaptive_smoothing import get_lambdas
from myhelpers.tripletloss import get_tripletLossLoader, get_triplet_criterion
from myhelpers.iterator import Infinite_iter
from myhelpers.memory import get_cuda_memory
from myhelpers.create_conv_layer import get_conv
from myhelpers.orthogonal_convolutions import deconv_orth_dist, deconv_orth_dist2
from myhelpers.imbalanced import get_class_weights


modelFinalCheckpoint = 'finalModel.pt'
modelTripletFinalCheckpoint = 'finalModel_triplet.pt'
modelStartCheckpoint = 'initModel.pt'
modelTripletStartCheckpoint = 'initModel_triplet.pt'

tripletModelFinalCheckpoint = 'finalModel.pt'

saved_models_per_iteration_folder= "iterations"
saved_models_per_iteration_name="iteration{0}.pt"
saved_models_triplet_per_iteration_name="iteration_triplet{0}.pt"

statsFileName = "stats.csv"
adaptiveSmoothingFileName = "adaptive_smoothing.csv"
timeFileName = "time.csv"
epochsFileName = "epochs.csv"

paramsFileName="params.json"

WANDB_message="wandb not working"

saved_models_per_iteration_frequency = 1

class ZeroModule(Module):
    def __init__(self, *args, **kwargs):
        super(torch.nn.Identity, self).__init__()

    def forward(self, input):
        return torch.zeros_like(input)

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# Create an FC layer with RELU and BatchNormalization
def get_fc(num_of_inputs, num_of_outputs, num_of_layers = 1, bnorm=False, relu=False):
    l = [] 
    
    for i in range(num_of_layers):
        n_out = num_of_inputs if (i+1 != num_of_layers) else num_of_outputs
        l.append(('linear'+str(i), torch.nn.Linear(num_of_inputs, n_out)))
        if bnorm == True:
            l.append(('bnorm'+str(i), torch.nn.BatchNorm1d(n_out)))
        if relu == True:
            l.append(('relu'+str(i), torch.nn.ReLU()))
        
    d = collections.OrderedDict(l)
    seq = torch.nn.Sequential(d)
    
    return seq

def create_pretrained_model(params):
    tl_model = params["tl_model"]
    inpt_size = params["img_res"]
    pretrained = params["pretrained"]
    tl_extralayer = params["tl_extralayer"]
    
    if tl_model == "CIFAR":
        #TODO: bring this back
        model = cifar_resnet56(pretrained='cifar100' if pretrained else None) # 0.693
        # model = cifar100(128, pretrained=True) # acc = 0.589
    elif tl_model == "ResNet18":
        model = models.resnet18(pretrained=pretrained)
        if tl_extralayer:
            model.layer4 = nn.Sequential(collections.OrderedDict([
                    ('layer4_1', model.layer4),
                    ('layer4_2', model._make_layer(models.resnet.BasicBlock, 1024, 2, stride=2)),
                ])) 

    elif tl_model == "ResNet50":
        model = models.resnet50(pretrained=pretrained)
    elif tl_model == "ResNet56":
        if pretrained:
            raise Exception('Cannot find pretrained ResNet56')
        model = resnet56()
    elif tl_model == "preResNet":
        model = preresnet_cifar(dataset='cifar100', inpt_size=inpt_size, pretrained=pretrained if pretrained==True else None)
    else:
        raise Exception('Unknown network type')
    
    try:
        num_ftrs = model.fc.in_features if not (tl_model == "ResNet18" and tl_extralayer) else 1024
    except:
        num_ftrs = model.classifier[0].in_features
        
    return model, num_ftrs

def parse_phyloDistances(phyloDistances_string):
    phyloDistances_list_string = phyloDistances_string.split(",")
    return list(map(lambda x: float(x), phyloDistances_list_string))


def get_architecture(params, csv_processor):
    modelType = params["modelType"]
    phylo_distances = parse_phyloDistances(params["phyloDistances"])
    num_of_fine = len(csv_processor.getFineList())
    architecture = { "fine": num_of_fine,}
    if modelType != "PhyloNN":
        architecture["coarse"] = len(csv_processor.getCoarseList())
    else:
        for i in phylo_distances:
            architecture[str(i).replace(".", "")+"distance"] = num_of_fine

    # print('architecture',architecture)
    return architecture



def create_model(architecture, params, device=None):
    model = None

    if params["modelType"] == "PhyloNN":
        model = CNN_PhyloNN(architecture, params, device=device)
    elif params["modelType"] == "exRandomFeatures":
        model = CNN_Predictor_Descriminator(architecture, params, device=device)
    elif params["modelType"] != "BB":
        model = CNN_Two_Nets(architecture, params, device=device)
    else:  
        model = CNN_One_Net(architecture, params, device=device)
        if params["tripletEnabled"]:
            model = CNN_One_Net_Triplet_Wrapper(model)

    if device is not None:
        model = model.to(device=device)
    else:
        print("Warning! model is on cpu")
    return model


# Gets a customized transfer learning model between two layers.
# from_layer=None => from first layer. Otherwise, pass layer name
# to_layer=None => to last layer. Otherwise, pass layer name
# Returns [from_layer, to_layer). i.e. t_layer not included.
def getCustomTL_layer(pretrained_model, from_layer=None, to_layer=None):
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


class CNN_PhyloNN(nn.Module):  
    # Contructor
    def __init__(self, architecture, params, device=None):

        modelType = params["modelType"]
        self.modelType = modelType
        # img_res = params["img_res"]
        self.phylo_distances = parse_phyloDistances(params["phyloDistances"])
        self.device=device

        if device is None:
            print("Creating model on cpu!")

        fc_layers = params["fc_layers"]
        
        super(CNN_PhyloNN, self).__init__()

        # The pretrained models
        self.network, num_ftrs = create_pretrained_model(params) 
        self.network = torch.nn.Sequential(*getCustomTL_layer(self.network, None, None)) # includes average pooling
        self.network = torch.nn.Sequential(*self.network, Flatten())
        # features = self.network(torch.rand(1, 3, img_res, img_res))
        # print('features', features.shape)
        # self.len_features = features.shape[1]
        self.len_features = num_ftrs
        
        output_size = architecture['fine']
        self.fc_layers = {'fine': get_fc(self.len_features, output_size, num_of_layers=fc_layers)}
        for i in self.phylo_distances:
            level_name = str(i).replace(".", "")+"distance"
            output_size = architecture[level_name]
            self.fc_layers[level_name] = get_fc(int(i*self.len_features), output_size, num_of_layers=fc_layers)
        self.fc_layers = torch.nn.ModuleDict(self.fc_layers)

        self.layer4_features = None
        def getLayer4Features(module, input, output):
            self.layer4_features = output
        self.network[7].register_forward_hook(getLayer4Features)

        if device is not None:
            self.fc_layers = self.fc_layers.cuda()
            self.network = self.network.cuda()
        
    def get_module(self,name):
        if name == 'layer1':
            return self.network[4]
        elif name == 'layer2':
            return self.network[5]
        elif name == 'layer3':
            return self.network[6]
        elif name == 'layer4':
            return self.network[7]
        elif name == 'fc':
            return self.fc_layers['fine'].linear0
        elif name == 'conv1':
            return self.network[0]
        else:
            raise Exception('layer not found')  
    
    # Prediction
    def forward(self, x):
        activations = self.activations(x)
        result = {
            "fine": activations["fine"]
        }
        for i in self.phylo_distances:
            level_name = str(i).replace(".", "")+"distance"
            result[level_name] = activations[level_name]
        # print('result', result)
        return result

    default_outputs = {
        "fine": True,
    }
    def activations(self, x, outputs=default_outputs):  
        self.layer4_features = None

        # print('x', x.shape)
        features = self.network(x)

        activations = {
            "input": x,
            "gap_features": features,
            "layer4_features": self.layer4_features,
            "fine": self.fc_layers['fine'](features) if outputs["fine"] else None
        }

        for i in self.phylo_distances:
            level_name = str(i).replace(".", "")+"distance"
            # print('level_name', level_name)
            # print('bbb',features[:, :int(self.len_features*i)].shape)
            # print('ccc', self.fc_layers[level_name])
            activations[level_name] = self.fc_layers[level_name](features[:, :int(self.len_features*i)])
        
        # print('activations', activations)
        return activations

class CNN_One_Net(nn.Module):
    def __init__(self, architecture, params, device=None):
        fc_layers = params["fc_layers"]
        img_res = params["img_res"]
        self.img_res = img_res

        self.numberOfFine = architecture["fine"]
        self.device=device

        if device is None:
            print("Creating model on cpu!")
        
        super(CNN_One_Net, self).__init__()

        # The pretrained models
        self.pretrained, num_ftrs_fine = create_pretrained_model(params)
        
        try:
            if self.numberOfFine != self.pretrained.fc.out_features:
                self.pretrained.fc = get_fc(num_ftrs_fine, self.numberOfFine, num_of_layers=fc_layers)
                # print('fc replaced!')
        except:
            if self.numberOfFine != self.pretrained.classifier[0].out_features:
                self.pretrained.classifier[0] = get_fc(num_ftrs_fine, self.numberOfFine, num_of_layers=fc_layers)
                # print('fc replaced!')

        self.intermediate_output_layers = ['layer1','layer2','layer3','layer4']
        for layer_ in self.intermediate_output_layers:
            try:
                layer = self.get_module(layer_)
                layer.register_forward_hook(self.get_activation(layer_))
            except:
                print(layer_, 'not found')

        if device is not None:
            self.pretrained = self.pretrained.cuda()

    def get_module(self,name):
        if name == 'layer1':
            return self.pretrained.layer1
        elif name == 'layer2':
            return self.pretrained.layer2
        elif name == 'layer3':
            return self.pretrained.layer3
        elif name == 'layer4':
            return self.pretrained.layer4
        else:
            raise Exception('layer not found')
    
    def get_activation(self, name):
        def hook(model, input, output, detach=False):
            self.activation[name] = output.detach() if detach==True else output
        return hook
    
    # Prediction
    def forward(self, x):
        return self.activations(x)


    def get_coarse(self, x, dataset):
        result = self(x)
        fineToCoarse = dataset.csv_processor.getFineToCoarseMatrix()
        if self.device is not None:
            fineToCoarse = fineToCoarse.cuda()
        result = torch.mm(result['fine'], fineToCoarse)
        return result


    default_outputs = {
        "fine": True,
        "coarse" : False
    }
    def activations(self, x, outputs=default_outputs):  

        self.activation = {
            "input": x,
            "coarse": None,
        }

        y = self.pretrained(x)

        self.activation['fine'] = y
        activation = self.activation
        self.activation = None

        return activation

class CNN_One_Net_Triplet_Wrapper(nn.Module):
    def __init__(self, network_fine):
        super(CNN_One_Net_Triplet_Wrapper, self).__init__()

        self.network_fine = network_fine
    
        tripletLossSpace_dim = self.network_fine.numberOfFine
        self.intermediate_outputs = {}

        # with torch.set_grad_enabled(False):
        with torch.no_grad():
            rand_input = torch.rand(2, 3, self.network_fine.img_res, self.network_fine.img_res)
            if self.network_fine.device is not None:
                rand_input = rand_input.cuda()
                
            isTraining=self.network_fine.training
            if isTraining:
                self.network_fine.eval()
            out_ = self.network_fine(rand_input)
            if isTraining:
                self.network_fine.train()

        for c in self.network_fine.intermediate_output_layers:
            if c in out_:
                features = out_[c]
                num_of_inputs = torch.flatten(features[1:]).shape[0]
                k = nn.Sequential(collections.OrderedDict([
                    ('flatten__'+str(c), Flatten()),
                    ('linear__'+str(c), nn.Linear(num_of_inputs, tripletLossSpace_dim)),
                    # ('bnorm__'+str(c), nn.BatchNorm1d(tripletLossSpace_dim)),
                    # ('relu__'+str(c), nn.ReLU()),
                ]))
                self.intermediate_outputs[c] = k
        self.intermediate_outputs = torch.nn.ModuleDict(self.intermediate_outputs)

        if self.network_fine.device is not None:
            for intermediate_output_name in self.intermediate_outputs:
                intermediate_output = self.intermediate_outputs[intermediate_output_name]
                intermediate_output = intermediate_output.cuda()
    
    
    def get_module(self,name):
        if name == 'layer1':
            return self.network_fine.layer1
        elif name == 'layer2':
            return self.network_fine.layer2
        elif name == 'layer3':
            return self.network_fine.layer3
        elif name == 'layer4':
            return self.network_fine.layer4
        else:
            raise Exception('layer not found')        

    
    
    # Prediction
    def forward(self, x):
        activations = self.network_fine(x)

        for intermediate_output_name in self.intermediate_outputs:
            intermediate_output = self.intermediate_outputs[intermediate_output_name]
            activations_sub = activations[intermediate_output_name]
            activations[intermediate_output_name] = intermediate_output(activations_sub)

        return activations


# Build Jie's network (predictor/descriminator)
class CNN_Predictor_Descriminator(nn.Module):  
    # Contructor
    def __init__(self, architecture, params, device=None):

        modelType = params["modelType"]
        self.modelType = modelType
        link_layer = params["link_layer"]
        self.numberOfFine = architecture["fine"]
        # self.numberOfCoarse = architecture["coarse"] if not modelType=="DSN" else architecture["fine"]
        self.device=device

        if device is None:
            print("Creating model on cpu!")

        fc_layers = params["fc_layers"]
        # img_res = params["img_res"]
        
        super(CNN_Predictor_Descriminator, self).__init__()

        # The pretrained models
        self.network_descriminator, num_ftrs_descriminator = create_pretrained_model(params)
        self.network_predictor, num_ftrs_predictor = create_pretrained_model(params)
        self.predictor_pre = torch.nn.Sequential(*getCustomTL_layer(self.network_predictor, None, link_layer)) 
        self.predictor_post = torch.nn.Sequential(*getCustomTL_layer(self.network_predictor, link_layer, None)) 
        self.predictor_post = torch.nn.Sequential(*self.predictor_post, Flatten())
        self.predictor_fc = get_fc(num_ftrs_predictor, self.numberOfFine, num_of_layers=fc_layers)
        
        # descriminator
        self.descriminator = torch.nn.Sequential(*getCustomTL_layer(self.network_descriminator, link_layer, None)) 
        self.descriminator = torch.nn.Sequential(*self.descriminator, Flatten())
        self.descriminator_fc = get_fc(num_ftrs_descriminator, self.numberOfFine, num_of_layers=fc_layers)

        if device is not None:
            self.descriminator_fc = self.descriminator_fc.cuda()
            self.descriminator = self.descriminator.cuda()

            self.predictor_pre = self.predictor_pre.cuda()
            self.predictor_post = self.predictor_post.cuda()
            self.predictor_fc = self.predictor_fc.cuda()
    
    def _freeze_internal(self, block, freeze=False):
        # print('freezing', self.predictor_pre)
        for (name, module) in block.named_children():
            # print(name)
            for layer in module.children():
                # print(layer)
                for param in layer.parameters():
                    param.requires_grad = freeze

    def freeze(self, discriminator=True, pre_predictor=True):
        self._freeze_internal(self.predictor_pre, pre_predictor)
        self._freeze_internal(self.descriminator, discriminator)
        self._freeze_internal(self.descriminator_fc, discriminator)
    
        
    def get_module(self,name):
        raise Exception('Not implemented yet!')   
    
    # Prediction
    def forward(self, x):
        activations = self.activations(x)
        result = {
            "fine": activations["fine"],
            "discriminator" : activations["discriminator"]
        }
        return result

    default_outputs = {
        "fine": True,
        "discriminator" : True
    }
    def activations(self, x, outputs=default_outputs):  
        # print(x.shape)
        features = self.predictor_pre(x)
        predictor_features = self.predictor_post(features)
        discriminator_features = self.descriminator(features)

        activations = {
            "input": x,
            "features": features,
            "predictor_features": predictor_features,
            "discriminator_features": discriminator_features,
            "discriminator": self.descriminator_fc(discriminator_features) if outputs["discriminator"] else None,
            "fine": self.predictor_fc(predictor_features) if outputs["fine"] else None
        }

        return activations

# Build a Hierarchical convolutional Neural Network with conv layers
class CNN_Two_Nets(nn.Module):  
    # Contructor
    def __init__(self, architecture, params, device=None):
        modelType = params["modelType"]
        self.modelType = modelType
        self.numberOfFine = architecture["fine"]
        self.numberOfCoarse = architecture["coarse"] if not (modelType=="DSN" or modelType=="PhyloNN") else architecture["fine"]
        self.device=device

        if device is None:
            print("Creating model on cpu!")

        link_layer = params["link_layer"]
        fc_layers = params["fc_layers"]
        img_res = params["img_res"]
        
        super(CNN_Two_Nets, self).__init__()

        # The pretrained models
        self.network_coarse, num_ftrs_coarse = create_pretrained_model(params)
        self.network_fine, num_ftrs_fine = create_pretrained_model(params)
        
        # h_y block
        self.h_y = torch.nn.Sequential(*getCustomTL_layer(self.network_coarse, None, link_layer)) 

        # print('hy')
        # summary(self.h_y.cuda(), (3, 32, 32))

        # g_c block
        self.g_c = None
        if modelType == "HGNNgcI":
            # TODO: fix this case. broken because we need avg pooling from h_y
            print("HGNNgcI case not supported yet")
            raise 
            self.g_c = [torch.nn.Identity()]
        elif modelType != "HGNNgc0" and modelType != "BB":
            self.g_c = getCustomTL_layer(self.network_coarse, link_layer, None)
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
            self.h_b = torch.nn.Sequential(*getCustomTL_layer(self.network_fine, None, link_layer))

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
                if self.device is not None:
                    self.cat_conv2d = self.cat_conv2d.cuda()

        # g_y block
        self.g_y = torch.nn.Sequential(*getCustomTL_layer(self.network_fine, link_layer, None),  
                                       Flatten())
        self.g_y_fc = get_fc(num_ftrs_fine, self.numberOfFine, num_of_layers=fc_layers)

        self.layer4_features = None
        def getLayer4Features(module, input, output):
            self.layer4_features = output
        self.network_fine.layer4.register_forward_hook(getLayer4Features)

        if device is not None:
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


    def get_coarse(self, x, dataset):
        result = self(x)
        if self.modelType!="DSN" and self.modelType!="BB" and self.modelType != "HGNNgc0":
            result = result['coarse']
        else:
            fineToCoarse = dataset.csv_processor.getFineToCoarseMatrix()
            if self.device is not None:
                fineToCoarse = fineToCoarse.cuda()
            result = torch.mm(result['fine'], fineToCoarse)
        return result

        
    def get_module(self,name):
        if name == 'layer1':
            return self.network_fine.layer1
        elif name == 'layer2':
            return self.network_fine.layer2
        elif name == 'layer3':
            return self.network_fine.layer3
        elif name == 'layer4':
            return self.network_fine.layer4
        elif name == 'fc':
            return self.g_y_fc.linear0
        elif name == 'conv1':
            return self.network_fine.conv1
        else:
            raise Exception('layer not found')   

    default_outputs = {
        "fine": True,
        "coarse" : True
    }
    def activations(self, x, outputs=default_outputs): 
        self.layer4_features = None

        # print(x.shape)
        hy_features = self.h_y(x)
        # print(hy_features.shape)
        
        hb_hy_features = None
        hb_features = None
        if self.h_b is not None:
            hy_features_feed = hy_features

            hb_features = self.h_b(x)
            if self.modelType == "HGNN":
                hb_hy_features = torch.cat((hy_features_feed, hb_features), 1)
                hb_hy_features =  self.cat_conv2d(hb_hy_features)
            elif self.modelType == "HGNN_cat":
                hb_hy_features = torch.cat((hy_features_feed, hb_features), 2)
            else:
                hb_hy_features = hy_features_feed + hb_features

        else:
            hb_hy_features = hy_features

        # print('hb_hy_features', hb_hy_features.shape)

        yc = None
        gc_features = None
        if outputs["coarse"] and self.g_c is not None:
            gc_features = self.g_c(hy_features)
            yc = self.g_c_fc(gc_features)
        
        y = None
        gy_features = None
        if outputs["fine"]:
            gy_features = self.g_y(hb_hy_features)
            # print('gy_features', gy_features.shape)
            y = self.g_y_fc(gy_features)
            # print('y', y.shape)
            

        modelType_has_coarse = gc_features is not None and (self.modelType!="DSN")  

        activations = {
            "input": x,
            "hy_features": hy_features, #mid genus
            "hb_features": hb_features, #mid species
            "hb_hy_features": hb_hy_features, #concat
            "gy_features": gy_features if outputs["fine"] else None, #final species
            "gc_features": gc_features if outputs["coarse"] else None, #final genus
            "coarse": yc if outputs["coarse"] and modelType_has_coarse else None,
            "fine": y if outputs["fine"] else None,
            "layer4_features": self.layer4_features
        }

        return activations

def getModelFile(experimentName, triplet=False):
    return os.path.join(experimentName, modelFinalCheckpoint if not triplet else modelTripletFinalCheckpoint)

def getInitModelFile(experimentName, triplet=False):
    return os.path.join(experimentName, modelStartCheckpoint if not triplet else modelTripletStartCheckpoint)

def f1_criterion(device):
    return lambda pred, true : torch.reciprocal(torch.tensor(get_f1(*getPredictions(pred, true), device=device), requires_grad=True))


def get_optimizer(optimizer_type, model, learning_rate, weight_decay):
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),momentum=0.9, nesterov=True, lr = learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adabelief':
        optimizer = AdaBelief(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)
        # optimizer = AdaBelief(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8, betas=(0.9,0.999), weight_decouple = (scheduler_type!="cifar"), rectify = False)
    else:
        raise Exception("Unknown optimizer")

    return optimizer

def get_scheduler(scheduler_type, optimizer, scheduler_gamma, scheduler_patience, learning_rate):
    if scheduler_type == "cifar":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150])
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_patience, gamma=scheduler_gamma)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_patience, eta_min=learning_rate*scheduler_gamma)
    elif scheduler_type == "plateau":    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=scheduler_gamma, patience=scheduler_patience)
    elif scheduler_type != "noscheduler": 
        raise Exception("scheduler type not found")
    
    return scheduler

def get_criterion(use_phylogeny_loss, phylogeny_loss_epsilon, distance_matrix, device, weight=None):
    weight_not_supported = True
    if not use_phylogeny_loss:
        criterion = nn.CrossEntropyLoss(weight=weight)
        # criterion = f1_criterion(device)
        # criterion = escort_criterion(device)
        
        weight_not_supported=False
    elif use_phylogeny_loss=="MSE":
        criterion = Phylogeny_MSE(distance_matrix, phylogeny_loss_epsilon)
    else:
        criterion = Phylogeny_KLDiv(distance_matrix)

    if device is not None:
        criterion = criterion.cuda()
    
    if weight_not_supported and (weight is not None):
        print("Warning! weighted", use_phylogeny_loss , "criterion is not supported")

    return criterion

def report_summaries(validation_loader, test_loader, model, params, device):
    if wandb.run is not None:
        predlist_val, lbllist_val = getLoaderPredictionProbabilities(validation_loader, model, params, device=device)
        predlist_val, lbllist_val = getPredictions(predlist_val, lbllist_val)
        validation_fine_f1 = get_f1(predlist_val, lbllist_val, device=device)

        predlist_test, lbllist_test = getLoaderPredictionProbabilities(test_loader, model, params, device=device)
        predlist_test, lbllist_test = getPredictions(predlist_test, lbllist_test)
        test_fine_f1 = get_f1(predlist_test, lbllist_test, device=device)
        test_fine_acc = get_f1(predlist_test, lbllist_test, device=device, acc=True)
    
        wandb.run.summary["validation_fine_f1"] = validation_fine_f1
        wandb.run.summary["test_fine_f1"] = test_fine_f1
        wandb.run.summary["test_fine_acc"] = test_fine_acc
#         wandb.run.summary.update({"final_logits": wandb.Histogram(logits)})

def get_metrics(train_loader, validation_loader, test_loader, model, params, device, isDSN, getCoarse, detailed_reporting):
    test_fine_f1 = None
    test_fine_acc = None
    test_coarse_f1 = None
    validation_coarse_loss = None
    validation_coarse_f1 = None
    training_coarse_loss = None
    training_coarse_f1 = None

    if test_loader and detailed_reporting:
        predlist_test, lbllist_test = getLoaderPredictionProbabilities(test_loader, model, params, device=device)
        predlist_test, lbllist_test = getPredictions(predlist_test, lbllist_test)
        test_fine_f1 = get_f1(predlist_test, lbllist_test, device=device)
        test_fine_acc = get_f1(predlist_test, lbllist_test, device=device, acc=True)
        if (not isDSN):
            predlist_test, lbllist_test = getLoaderPredictionProbabilities(test_loader, model, params, 'coarse', device=device)
            predlist_test, lbllist_test = getPredictions(predlist_test, lbllist_test)
            test_coarse_f1 = get_f1(predlist_test, lbllist_test, device=device)
    
    predlist_val, lbllist_val = getLoaderPredictionProbabilities(validation_loader, model, params, device=device)
    validation_loss = getCrossEntropy(predlist_val, lbllist_val)
    predlist_val, lbllist_val = getPredictions(predlist_val, lbllist_val)
    validation_fine_f1 = get_f1(predlist_val, lbllist_val, device=device)

    if detailed_reporting and not isDSN:
        predlist_val, lbllist_val = getLoaderPredictionProbabilities(validation_loader, model, params, 'coarse', device=device)
        if getCoarse:
            validation_coarse_loss = getCrossEntropy(predlist_val, lbllist_val)
            predlist_val, lbllist_val = getPredictions(predlist_val, lbllist_val)
            validation_coarse_f1 = get_f1(predlist_val, lbllist_val, device=device)

    predlist_train, lbllist_train = getLoaderPredictionProbabilities(train_loader, model, params, device=device)
    training_loss = getCrossEntropy(predlist_train, lbllist_train)
    predlist_train, lbllist_train = getPredictions(predlist_train, lbllist_train)
    training_fine_f1 = get_f1(predlist_train, lbllist_train, device=device)
    training_fine_acc = get_f1(predlist_train, lbllist_train, device=device, acc=True)

    if detailed_reporting and not isDSN:
        predlist_train, lbllist_train = getLoaderPredictionProbabilities(train_loader, model, params, 'coarse', device=device)
        if getCoarse:
            training_coarse_loss = getCrossEntropy(predlist_train, lbllist_train)
            predlist_train, lbllist_train = getPredictions(predlist_train, lbllist_train)
            training_coarse_f1 = get_f1(predlist_train, lbllist_train, device=device)
    
    return {
        'test_fine_f1': test_fine_f1,
        'test_fine_acc': test_fine_acc,
        'test_coarse_f1': test_coarse_f1,
        'training_coarse_f1': training_coarse_f1,
        'training_coarse_loss': training_coarse_loss,
        'training_fine_acc': training_fine_acc,
        'training_fine_f1': training_fine_f1,
        'training_loss': training_loss,
        'validation_coarse_f1': validation_coarse_f1,
        'validation_loss': validation_loss,
        'validation_coarse_loss': validation_coarse_loss,
        'validation_fine_f1': validation_fine_f1
    }

def save_experiment(model, df, df_adaptive_smoothing, time_elapsed, savedModelName, epochs, params, detailed_reporting):
    # save model
    torch.save(model.state_dict(), os.path.join(savedModelName, modelFinalCheckpoint))
    # save results
    df.to_csv(os.path.join(savedModelName, statsFileName))  

    if detailed_reporting:
        df_adaptive_smoothing.to_csv(os.path.join(savedModelName, adaptiveSmoothingFileName))  
    
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

def add_layer3_L1loss(loss_name, output, losses):
    o = output
    w = o.shape[2]
    m = torch.nn.MaxPool2d(w)
    o2 = m(o).squeeze()
    max_, _ = torch.max(o2, dim=1)
    o2 = o2/max_.reshape(-1, 1)
    o2 = torch.norm(o2, dim=1)
    o2 = torch.mean(o2)
    losses[loss_name] = o2

def add_loss(loss_name, criterion, output, batch, isDSN, losses):
    batch_lossname = loss_name if (loss_name == 'fine') or isDSN else 'coarse'
    losses[loss_name] = criterion(output[loss_name], batch[batch_lossname])

def add_triplet_losses(loss_name, z_triplet, batch_triplet, params, nonzerotriplets, csv_processor, losses, detailed_reporting, triplet_hists, device):
    margin = params["tripletMargin"]
    regularTripletLoss = params["regularTripletLoss"]
    selection_criterion = params["tripletSelector"]
    triplet_layers_dic = params["triplet_layers_dic"].split(',')
    if detailed_reporting:
        print('triplet layers', triplet_layers_dic)
    triplet_criterion = get_triplet_criterion(margin, selection_criterion, not regularTripletLoss, triplet_layers_dic, device)

    # get loss from criterion 
    if (loss_name in z_triplet) and z_triplet[loss_name] is not None:
        # Check if this layer has a triplet loss implemented
        if csv_processor.get_target_from_layerName(batch_triplet, loss_name, not regularTripletLoss, z_triplet, triplet_layers_dic) is not None:
            losses[loss_name], nonzerotriplets[loss_name] = triplet_criterion(z_triplet, batch_triplet, loss_name, csv_processor)

            if selection_criterion=="semihard":  
                # print(losses[loss_name])
                assert (losses[loss_name]-margin).le(torch.zeros_like(losses[loss_name])).all()
            
            if detailed_reporting:
                z_triplet_layer = z_triplet[loss_name]
                if device is not None:
                    z_triplet_layer = z_triplet_layer.detach().cpu()
                # z_triplet_layer_norm = torch.norm(z_triplet_layer, dim=1) 
                z_triplet_layer_norm = z_triplet_layer
                z_triplet_layer_norm = z_triplet_layer_norm.numpy()
                triplet_hists[loss_name].append(z_triplet_layer_norm)

class Random_label_loss():
    def __init__(self):
        self.random_label_loss = None

    def add_discriminator_predictor_loss(self, loss_name, z, batch, criterion, fine_list, losses, device):
        if (self.random_label_loss is None):
            numOfFine = len(fine_list)
            map_to_random_label =  map(lambda x : random.choice([i for i in range(0, numOfFine) if i not in [x]]), batch['fine'])
            batch_random_labels = torch.LongTensor(list(map_to_random_label)).view(-1)
            if device:
                batch_random_labels = batch_random_labels.to(device=device)
            diff = batch['fine'] - batch_random_labels
            num_of_notRandom = (diff == 0.).sum(dim=0)# torch.count_nonzero(batch['fine'] - batch_random_labels)
            # print("Is there any non random label?", num_of_notRandom, num_of_notRandom > 0)
            assert(num_of_notRandom == 0)
            self.random_label_loss = criterion(z['discriminator'], batch_random_labels)
        
        if loss_name == 'discriminator':
            losses[loss_name] = self.random_label_loss
        elif loss_name == 'negative_discriminator':  
            losses[loss_name] = 1.0/self.random_label_loss


class Species_distance_sisters():
    # Contructor
    def __init__(self, csv_processor, genetic_distances):
        self.map = {}
        self.csv_processor = csv_processor
        for species in csv_processor.tax.node_ids:
            self.map[species] = {}
            for distance in genetic_distances:
                self.map[species][str(distance).replace(".", "")+"distance"] = csv_processor.tax.get_siblings_by_name(species, distance)

        # print('self.map', self.map)

    def map_speciesId_sisterVector(self, speciesId, loss_name):
        fine_list = self.csv_processor.getFineList()
        species = fine_list[speciesId]
        sisters = self.map[species][loss_name]
        sisters_indices = list(map(lambda x: fine_list.index(x), sisters))
        # print('sisters0', loss_name, speciesId, species, sisters, sisters_indices, range(len(fine_list)))
        return sisters_indices

def add_phyloNN_losses(loss_name, z, batch, criterion, layers, sisters_object, losses, device):
    if loss_name == 'fine':
        return

    true_fine = batch['fine']
    if loss_name in layers:
        layer_truth = list(map(lambda x: sisters_object.map_speciesId_sisterVector(x, loss_name), true_fine))
        mlb = MultiLabelBinarizer(range(len(sisters_object.csv_processor.getFineList())))
        hotcoded_sisterindices = torch.FloatTensor(mlb.fit_transform(layer_truth))
        if device:
            hotcoded_sisterindices = hotcoded_sisterindices.to(device=device)
        # print('layer_truth', loss_name, true_fine, layer_truth, hotcoded_sisterindices, z[loss_name], hotcoded_sisterindices.shape, z[loss_name].shape)
        losses[loss_name] = criterion(z[loss_name], hotcoded_sisterindices)
    # print('add_phyloNN_losses', losses)

def add_intraKernelOrthogonalityLoss(loss_name, model, phyloDistances, losses):
    # get conv layer
    conv_layer = model.network[7][1].conv2
    conv_layer_weights = conv_layer.weight

    # get corresponding output convolutions
    loss = 0
    abs_total_dist = conv_layer.weight.shape[1]
    phyloDistancesPlus = [1] + phyloDistances + [0]
    # print(phyloDistancesPlus)
    for distance_indx, element in enumerate(phyloDistancesPlus):
        if distance_indx == len(phyloDistancesPlus)-1: break
        distance = int(phyloDistancesPlus[distance_indx]*abs_total_dist)
        prev_distance = int(phyloDistancesPlus[distance_indx+1]*abs_total_dist)
        conv_layer_sub = conv_layer_weights[:, prev_distance:distance, :, :]
        # print(prev_distance, distance)
        for distance_indx2, element2 in enumerate(phyloDistancesPlus):
            if distance_indx2<= distance_indx: continue
            if distance_indx2 == len(phyloDistancesPlus)-1: break
            distance2 = int(phyloDistancesPlus[distance_indx2]*abs_total_dist)
            prev_distance2 = int(phyloDistancesPlus[distance_indx2+1]*abs_total_dist)
            conv_layer_sub2 = conv_layer_weights[:, prev_distance2:distance2, :, :]
            # print(prev_distance2, distance2)

            # get loss for their orthogonality
            loss = loss + deconv_orth_dist2(conv_layer_sub, conv_layer_sub2, stride = conv_layer.stride, padding = conv_layer.padding)
            # print(loss)

    losses[loss_name] = loss

def add_KernelOrthogonalityLoss(loss_name, model, phyloDistances, losses):
    # get conv layer
    # print('model debug', model.network)
    # print(model.network[4])
    # print(model.network[4][1])
    # print(model.network[4][1].conv2)
    # print(model.network[4][1].conv2.weight)
    conv_layer = model.network[7][1].conv2
    conv_layer_weights = conv_layer.weight

    # get corresponding output convolutions
    abs_total_dist = conv_layer.weight.shape[1]
    if loss_name!="1KernelOrthogonality":
        # print(loss_name)
        distance_indx = [idx for idx, element in enumerate(phyloDistances) if loss_name == str(element).replace(".", "")+"KernelOrthogonality"][0]
        distance = phyloDistances[distance_indx]
        prev_distance = phyloDistances[distance_indx+1] if distance_indx<len(phyloDistances)-1 else 0
    else:
        prev_distance = phyloDistances[0]
        distance = 1
    
    prev_distance = int(prev_distance*abs_total_dist)
    distance = int(distance*abs_total_dist)
    conv_layer_sub = conv_layer_weights[:, prev_distance:distance, :, :]
    # print('distance_indx', prev_distance, distance, conv_layer_sub.shape)

        
    # get loss for their orthogonality
    loss = deconv_orth_dist(conv_layer_sub, stride = conv_layer.stride, padding = conv_layer.padding)
    # print('loss', loss)
    losses[loss_name] = loss

def set_lambdas(params, layers):
    tripletEnabled = params["tripletEnabled"]
    modelType = params["modelType"]
    exRandomFeatures = (modelType == "exRandomFeatures")
    phyloNN = (modelType == "PhyloNN")
    two_phase_lambda = params["two_phase_lambda"]
    default_lambda = params["lambda"]
    L1_experiment = params["L1reg"]
    KernelOrthogonality = params["addKernelOrthogonality"]
    
    lambdas = {
        'fine': 1 if two_phase_lambda==False else default_lambda,
        'coarse': default_lambda if two_phase_lambda==False else 1,
    } 

    layersKeys = []
    if phyloNN:
        layersKeys.append('PhyloNN')
    if exRandomFeatures:
        layersKeys.append('predictor_descriminator_layers')
    if tripletEnabled:
        layersKeys.append('triplet_layers')
    if KernelOrthogonality:
        layersKeys.append('KernelOrthogonality')
        layersKeys.append('intraKernelOrthogonality')
    
    for layersKey in layersKeys:
        for layer in layers[layersKey]:
                if layer != 'fine':
                    lambdas[layer] = default_lambda if two_phase_lambda==False else 1
                else:
                    lambdas[layer] = 1 if two_phase_lambda==False else default_lambda

    if L1_experiment:
        lambdas["layer3"] = default_lambda
    
    # print('lambdas', lambdas)
    return lambdas

def flip_lambdas(lambdas, default_lambda, excluded_losses_from_lambda):
    for lambda_key in lambdas:
        lambdas[lambda_key] = default_lambda if lambda_key not in excluded_losses_from_lambda else 1

def get_lossnames(params, layers):
    modelType = params["modelType"]
    exRandomFeatures = (modelType == "exRandomFeatures")
    phyloNN = (modelType == "PhyloNN")
    tripletEnabled = params["tripletEnabled"]
    L1_experiment = params["L1reg"]
    KernelOrthogonality = params["addKernelOrthogonality"]
    if exRandomFeatures:
        loss_names = layers['predictor_descriminator_layers']
    elif phyloNN:
        loss_names = layers['PhyloNN']
    else:
        loss_names = layers['HGNN_layers']
    if tripletEnabled:
        loss_names = loss_names + layers['triplet_layers']
    if KernelOrthogonality:
        loss_names = loss_names + layers['KernelOrthogonality']
        loss_names = loss_names + layers['intraKernelOrthogonality']
    # L1 experiment
    if L1_experiment:
        loss_names = loss_names + layers['l1_layer']
    
    # print('loss_names', loss_names)
    return loss_names

def define_layerNames(params):
    layers = {
        'HGNN_layers': ['fine', 'coarse'],
        'predictor_descriminator_layers': ['fine', 'negative_discriminator', 'discriminator'],
        'triplet_layers': ['layer1','layer2','layer3','layer4'],
        'l1_layer': ['layer3']
    }
    
    layers['PhyloNN'] = ['fine']
    layers['KernelOrthogonality'] = ['1KernelOrthogonality']
    layers['intraKernelOrthogonality'] = ['intraKernelOrthogonality']
    KernelOrthogonality = params["addKernelOrthogonality"]
    modelType = params["modelType"]
    phyloNN = (modelType == "PhyloNN")
    
    if phyloNN:
        phylo_distances = parse_phyloDistances(params["phyloDistances"])
        for i in phylo_distances:
            level_name = str(i).replace(".", "")+"distance"
            layers['PhyloNN'].append(level_name)

            if KernelOrthogonality:
                level_name = str(i).replace(".", "")+"KernelOrthogonality"
                layers['KernelOrthogonality'].append(level_name)
        


    # print('layers',layers)
    return layers


def trainModel(train_loader, validation_loader, params, model, savedModelName, test_loader=None, device=None, detailed_reporting=False):      
    n_epochs = params["epochs"]
    patience = params["patience"] if params["patience"] > 0 else n_epochs
    learning_rate = params["learning_rate"]
    modelType = params["modelType"]
    two_phase_lambda = params["two_phase_lambda"]
    use_phylogeny_loss = params["phylogeny_loss"]
    phylogeny_loss_epsilon = params["phylogeny_loss_epsilon"]
    tripletEnabled = params["tripletEnabled"]
    scheduler_type = params["scheduler"]
    optimizer_type = params["optimizer"]
    weight_decay = params["weightdecay"]
    scheduler_patience = params["scheduler_patience"] if params["scheduler_patience"] > 0 else n_epochs
    scheduler_gamma = params["scheduler_gamma"]
    L1_experiment = params["L1reg"]
    exRandomFeatures = (modelType == "exRandomFeatures")
    phyloNN = (modelType == "PhyloNN")
    KernelOrthogonality = params["addKernelOrthogonality"]
    isBlackbox = (modelType == "BB")
    isDSN = (modelType == "DSN")
    phyloDistances = params["phyloDistances"]

    # Define layers for losses
    layers = define_layerNames(params)

    # Set lambdas
    lambdas = set_lambdas(params, layers)

    # Get triplet iterator
    if tripletEnabled:
        n_samples = params["tripletSamples"]
        triplets_train_loader = get_tripletLossLoader(train_loader.dataset, n_samples)
        # triplets_validation_loader = get_tripletLossLoader(validation_loader.dataset, n_samples, cuda=device)
        # triplets_test_loader = get_tripletLossLoader(test_loader.dataset, n_samples, cuda=device) if test_loader is not None else None
        triplet_train_iterator = Infinite_iter(triplets_train_loader)

    if device is None:
        print("training model on CPU!")
    
    # Setup adaptive smoothing
    adaptive_smoothing_enabled = params["adaptive_smoothing"]
    assert (not adaptive_smoothing_enabled) or (not two_phase_lambda), "Cannot have adaptive smoothing and two phase lambdas at the same time"
    adaptive_lambda = None if not adaptive_smoothing_enabled else params["adaptive_lambda"]
    adaptive_alpha = None if not adaptive_smoothing_enabled else params["adaptive_alpha"]
    df_adaptive_smoothing = pd.DataFrame()

    assert ((use_phylogeny_loss==False) or isBlackbox), "Coarse loss and phylogeny tree are not simultaneously supported." 
    assert ((KernelOrthogonality==False) or phyloNN), "Can't have Kernel orthogonality without PhyloNN architecture" 
    if phyloNN:
        phylo_distances_floats = parse_phyloDistances(phyloDistances)
        assert all(phylo_distances_floats[i] > phylo_distances_floats[i+1] for i in range(len(phylo_distances_floats)-1)), "phyloDistances should be ordered"
        assert all(phylo_distances_floats[i] > 0 and phylo_distances_floats[i] < 1 for i in range(len(phylo_distances_floats)-1)), "phyloDistances should be between 0 and 1"
    
    df = pd.DataFrame()
    
    # Create directories
    if not os.path.exists(savedModelName):
        os.makedirs(savedModelName)
    saved_models_per_iteration = os.path.join(savedModelName, saved_models_per_iteration_folder)
    if not os.path.exists(saved_models_per_iteration):
        os.makedirs(saved_models_per_iteration)

    # Setup hyperparameters
    optimizer = get_optimizer(optimizer_type, model, learning_rate, weight_decay)
    scheduler = get_scheduler(scheduler_type, optimizer, scheduler_gamma, scheduler_patience, learning_rate)
    criterion = get_criterion(use_phylogeny_loss, phylogeny_loss_epsilon, train_loader.dataset.csv_processor.distance_matrix, device, weight=get_class_weights(train_loader.dataset.get_labels()))
    early_stopping = EarlyStopping(path=savedModelName, patience=patience)

    if phyloNN:
        criterion_phyloNN = torch.nn.BCEWithLogitsLoss()
        sisters_object = Species_distance_sisters(train_loader.dataset.csv_processor, parse_phyloDistances(params["phyloDistances"]))

    print("Training started...")
    start = time.time()

    
    with tqdm(total=n_epochs, desc="iteration") as bar:
        epochs = 0
        # For each epoch
        for epoch in range(n_epochs):
            # print('epoch', epoch)

            if epoch != 0:
                triplet_hists = {}
                if tripletEnabled and detailed_reporting:
                    # Triplet loss histogram
                    for layer_name in layers['triplet_layers']:
                        triplet_hists[layer_name] = []

                labels_logged_fordataimbalance=[]            
                    
                model.train()
                # For each batch
                for i, batch in enumerate(train_loader):
                    # print('batch', i)
                    absolute_batch = i + epoch*len(train_loader)
        
                    if device is not None:
                        batch["image"] = batch["image"].cuda()
                        batch["fine"] = batch["fine"].cuda()
                        batch["coarse"] = batch["coarse"].cuda()
                        
                    labels_logged_fordataimbalance = labels_logged_fordataimbalance + batch["fine"].tolist()

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        # print('z before', get_cuda_memory(device))
                        z = applyModel(batch["image"], model)
                        
                        # Get loss names
                        loss_names = get_lossnames(params, layers)

                        losses = {}
                        nonzerotriplets = {}
                        adaptive_info = {
                            'batch': absolute_batch,
                            'epoch': epoch,
                        }

                        if tripletEnabled:
                            batch_triplet = triplet_train_iterator.getBatch()
                            # print('gotbatch', get_cuda_memory(device))
                            batch_triplet_input = batch_triplet['image']
                            if device is not None:
                                batch_triplet_input =batch_triplet_input.cuda()
                            # get output of model
                            z_triplet = applyModel(batch_triplet_input, model)

                        # Populate losses
                        if exRandomFeatures:
                            random_label_loss = Random_label_loss()
                        for loss_name in loss_names:
                            if loss_name=="layer3" and L1_experiment:
                                add_layer3_L1loss(loss_name, z[loss_name], losses)
                            
                            if loss_name in layers['HGNN_layers']:
                                if z[loss_name] is not None:
                                    add_loss(loss_name, criterion, z, batch, isDSN, losses)
                            elif tripletEnabled:
                                add_triplet_losses(loss_name, z_triplet, batch_triplet, params, nonzerotriplets, train_loader.dataset.csv_processor, losses, detailed_reporting, triplet_hists, device)
                            if (loss_name in layers['predictor_descriminator_layers']) and (exRandomFeatures):
                                fine_list = train_loader.dataset.csv_processor.getFineList()
                                random_label_loss.add_discriminator_predictor_loss(loss_name, z, batch, criterion, fine_list, losses, device)
                            if (loss_name in layers['PhyloNN']) and phyloNN:
                                add_phyloNN_losses(loss_name, z, batch, criterion_phyloNN, layers['PhyloNN'], sisters_object, losses, device)
                            if (loss_name in layers['KernelOrthogonality']) and KernelOrthogonality:
                                phyloDistances = parse_phyloDistances(params["phyloDistances"])
                                add_KernelOrthogonalityLoss(loss_name, model, phyloDistances, losses)
                            if (loss_name in layers['intraKernelOrthogonality']) and KernelOrthogonality:
                                add_intraKernelOrthogonalityLoss(loss_name, model, phyloDistances, losses)
                            
                            if loss_name in losses:
                                adaptive_info['loss_'+loss_name] = losses[loss_name].item() if torch.is_tensor(losses[loss_name]) else losses[loss_name]
                                adaptive_info['lambda_'+loss_name] = lambdas[loss_name] 
                                if loss_name in nonzerotriplets:
                                    adaptive_info['nonzerotriplets_'+loss_name] = nonzerotriplets[loss_name]  
                        
                        #Two phase training
                        excluded_losses_from_lambda = ['fine'] # , 'discriminator'
                        if two_phase_lambda and (epoch == int(n_epochs/2)):
                            flip_lambdas(lambdas, params['lambda'], excluded_losses_from_lambda)
                        # Adaptive smoothing 
                        elif adaptive_smoothing_enabled:
                            lambdas = get_lambdas(adaptive_alpha,   
                                adaptive_lambda, 
                                losses['fine'], 
                                {x: losses[x] for x in losses if x not in excluded_losses_from_lambda})
                            # if exRandomFeatures:
                            #     lambdas['discriminator'] = 1

                        df_adaptive_smoothing = df_adaptive_smoothing.append(pd.DataFrame(adaptive_info, index=[0]), ignore_index = True) 

                        # Add up losses
                        loss = 0
                        # if exRandomFeatures:
                        #     model.freeze(discriminator=True, pre_predictor=False)
                        # print('losses', losses)
                        for loss_name in losses:
                            # if loss_name != 'discriminator':
                            loss = loss + lambdas[loss_name]*losses[loss_name]

                        loss.backward()
                        optimizer.step()

                        # if exRandomFeatures:
                        #     optimizer.zero_grad()
                        #     model.freeze(discriminator=False, pre_predictor=True)
                        #     losses['discriminator'].backward()
                        #     optimizer.step()
                            
                        #     print('discriminator loss updated')

#                         print([train_loader.dataset.csv_processor.getFineList()[i] for i in labels_logged_fordataimbalance])
                        wandb_dic = {"loss": loss.item()}
                        wandb_dic = {**wandb_dic, **adaptive_info} 
                        try_running(lambda : wandb.log(wandb_dic), WANDB_message)
                    
                        # print('end_of_batch', get_cuda_memory(device))
                
                try_running(lambda : wandb.log({"data_balance_histogram": wandb.Histogram(labels_logged_fordataimbalance), 'epoch':epoch}), WANDB_message)
                        
                # Plot triplet loss histogram
                if tripletEnabled and detailed_reporting:
                    for layer_name in triplet_hists:
                        try_running(lambda : wandb.log({"triplet_norm_"+layer_name: wandb.Histogram(triplet_hists[layer_name]), 'epoch':epoch}), WANDB_message)
                
            model.eval()

            getCoarse = not isDSN and not isBlackbox and not exRandomFeatures
            metrics = get_metrics(train_loader, validation_loader, test_loader, model, params, device, isDSN, getCoarse, detailed_reporting)
            # print('metrics', metrics)

            # update scheduler
            if epoch != 0 and scheduler_type != "noscheduler":
                if scheduler_type != "plateau":
                    scheduler.step() 
                else:
                    scheduler.step(metrics['validation_fine_f1'])   
            
            # report epoch 
            row_information = {
                'epoch': epoch,
                'learning rate': optimizer.param_groups[0]['lr'],
            }
            row_information = {**row_information, **metrics}
            df = df.append(pd.DataFrame(row_information, index=[0]), ignore_index = True)
            try_running(lambda : wandb.log(row_information), WANDB_message)
            
            # Update the bar
            bar.set_postfix(val=row_information["validation_fine_f1"], 
                            train=row_information["training_fine_f1"],
                            val_loss=row_information["validation_loss"],
                            min_val_loss=early_stopping.val_loss_min)
            bar.update()

            # Update model
            if detailed_reporting and (epochs % saved_models_per_iteration_frequency == 0):
                model_name_path = os.path.join(savedModelName, saved_models_per_iteration_folder, saved_models_per_iteration_name).format(epochs)
                try:
                    torch.save(model.state_dict(),model_name_path)
                except:
                    print("model", model_name_path, "could not be updated!")
                    pass

            # early stopping
            if epoch != 0:
                early_stopping(1/row_information['validation_fine_f1'], epoch, model)

            epochs = epochs + 1
            if early_stopping.early_stop:
                print("Early stopping")
                print("total number of epochs: ", epoch)
                break
            
        
    # Register time
    end = time.time()
    time_elapsed = end - start
    
    # load the last checkpoint with the best model
    if epochs > 0:
        model.load_state_dict(early_stopping.getBestModel())
        
    
    # save information
    if savedModelName is not None:
        save_experiment(model, df, df_adaptive_smoothing, time_elapsed, savedModelName, epochs, params, detailed_reporting)

    report_summaries(validation_loader, test_loader, model, params, device)
    
    return df, epochs, time_elapsed

# loads a saved model along with its results
def loadModel(model, savedModelName, device=None):
    model.load_state_dict(torch.load(os.path.join(savedModelName, modelFinalCheckpoint), map_location=torch.device('cpu'))) 
    if device is not None:
        model.cuda()
    else:
        print("Model loaded into cpu!")

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
# Takes probabilities
def getAvgProbCorrectGuess(predlist, lbllist):
    lbllist = lbllist.reshape(lbllist.shape[0], -1)
    predlist = predlist.gather(1, lbllist)
    max_predlist = predlist.mean().item()
    return max_predlist

# # Returns the mean of best probability of all predictions. If low, it means the model is sure about its predictions
# def getAvgEntropyFromLoader(loader, model, params, label="fine"):
#     predlist, _ = getLoaderPredictionProbabilities(loader, model, params, label)
#     return torch.Tensor(entropy(predlist.cpu().T, base=2)).mean().item()

# Takes probabilities
def getCrossEntropy(predlist, lbllist):
    criterion = nn.CrossEntropyLoss()
    return criterion(predlist, lbllist).item()

def getLoaderPredictionProbabilities(loader, model, params, label="fine", device=None):
    if device is None:
        print("Warning! getLoaderPredictionProbabilities is on cpu")
    
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0)
    lbllist=torch.zeros(0, dtype=torch.long)
    
    if device is not None:
        predlist = predlist.cuda()
        lbllist = lbllist.cuda()

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in loader:
            if device is not None:
                batch["image"] = batch["image"].cuda()
                batch["fine"] = batch["fine"].cuda()
                batch["coarse"] = batch["coarse"].cuda()

            inputs = batch["image"]
            classes = batch[label]
            preds = applyModel(inputs, model)

            if label in preds and preds[label] is not None:
                preds = preds[label]
            elif label == 'coarse':
                fineToCoarseMatrix = loader.dataset.csv_processor.getFineToCoarseMatrix()
                if device is not None:
                    fineToCoarseMatrix = fineToCoarseMatrix.cuda()
                preds = torch.mm(preds['fine'],fineToCoarseMatrix) 
            else:
                raise

            preds = torch.nn.Softmax(dim=1)(preds)

            # Append batch prediction results
            predlist=torch.cat([predlist,preds], 0)
            lbllist=torch.cat([lbllist,classes], 0)  
            
    return predlist, lbllist


# Takes probabilities
def getPredictions(predlist, lbllist):
    _, predlist = torch.max(predlist, 1)  
        
    return predlist, lbllist

# Takes predictions
def get_f1(predlist, lbllist, device=None, acc=False):
    if device is not None:
        predlist = predlist.cpu()
        lbllist = lbllist.cpu()   
    if acc == False:
        return f1_score(lbllist, predlist, average='macro')
    return accuracy_score(lbllist, predlist)

# Returns the distance between examples in terms of classification cross entropy
# Augmentation should be disabled
def get_distance_from_example2(dataset_predProblist, example_1_predProb):
    criterion = torch.nn.CosineSimilarity()
    dataset_size = dataset_predProblist.shape[0]
    result = torch.zeros(1, dataset_size)

    with torch.set_grad_enabled(False):
        z_1 = example_1_predProb.reshape((1, example_1_predProb.shape[0]))
        z_1 = torch.nn.Softmax(dim=1)(z_1)
        for j in range(dataset_size):
            example_2_predProb = dataset_predProblist[j, :]

            z_2 = example_2_predProb.reshape((1, example_2_predProb.shape[0]))
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
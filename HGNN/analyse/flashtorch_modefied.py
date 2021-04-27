import warnings

import torch
import torch.nn as nn
import ntpath

defaultOutputs = {
    "fine": True,
    "coarse" : True
}


# A model wrapper used tointerface with torchflash 
class CNN_wrapper(torch.nn.Module):
    
    # Contructor
    def __init__(self, model, params, dataset):
                
        super(CNN_wrapper, self).__init__()
        self.model = model
        self.dataset= dataset
        self.setOutputsOfInterest({
            "fine": True,
            "coarse" : False
            })
    
    def setOutputsOfInterest(self, outputs):
        self.outputs = outputs
        
    # Prediction
    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()

        try:
            result = self.model.activations(x, defaultOutputs)
        except:
            result = self.model(x)

        if self.outputs['fine']:
            result = result['fine']
        elif self.outputs['coarse']:
            result =  self.model.get_coarse(x, self.dataset)
        return result
    
    
    
#################################################
    
from flashtorch.utils import (denormalize,
                              format_for_plotting,
                              standardize_and_clip)
    

def visualizeAllClasses(self, image_normalized, listOfClasses, guided=False, use_gpu=False):
    w = image_normalized.shape[3]
    h = image_normalized.shape[2]
    colors = image_normalized.shape[1]
    
    result = torch.zeros(1, colors, h, w*len(listOfClasses))
    idx = 0
    for i in listOfClasses:
        result[:, :, :, idx*w:(idx+1)*w] = self.calculate_gradients(image_normalized,
                                         i,
                                         guided=guided,
                                         take_max=True, #True
                                         use_gpu=use_gpu)
        idx = idx + 1
        
    stdized = standardize_and_clip(result,saturation=0.4)
    
    return stdized
    
#############################################################

class Backprop:
    """Provides an interface to perform backpropagation.
    This class provids a way to calculate the gradients of a target class
    output w.r.t. an input image, by performing a single backprobagation.
    The gradients obtained can be used to visualise an image-specific class
    saliency map, which can gives some intuition on regions within the input
    image that contribute the most (and least) to the corresponding output.
    More details on saliency maps: `Deep Inside Convolutional Networks:
    Visualising Image Classification Models and Saliency Maps
    <https://arxiv.org/pdf/1312.6034.pdf>`_.
    Args:
        model: A neural network model from `torchvision.models
            <https://pytorch.org/docs/stable/torchvision/models.html>`_.
    """

    ####################
    # Public interface #
    ####################

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.handle_forward=[]
        self.handle_backward=[]
        self.handle_conv=[]
        self.relu_registered = False
        self._register_conv_hook()

    def __del__(self): 
        for i in self.handle_forward:
            i.remove()
        for i in self.handle_backward:
            i.remove()
        for i in self.handle_conv:
            i.remove()
        self.gradients = None
        self.relu_outputs = {}

    def calculate_gradients(self,
                            input_,
                            target_class=None,
                            take_max=False,
                            guided=False,
                            use_gpu=False):

        """Calculates gradients of the target_class output w.r.t. an input_.
        The gradients is calculated for each colour channel. Then, the maximum
        gradients across colour channels is returned.
        Args:
            input_ (torch.Tensor): With shape :math:`(N, C, H, W)`.
            target_class (int, optional, default=None)
            take_max (bool, optional, default=False): If True, take the maximum
                gradients across colour channels for each pixel.
            guided (bool, optional, default=Fakse): If True, perform guided
                backpropagation. See `Striving for Simplicity: The All
                Convolutional Net <https://arxiv.org/pdf/1412.6806.pdf>`_.
            use_gpu (bool, optional, default=False): Use GPU if set to True and
                `torch.cuda.is_available()`.
        Returns:
            gradients (torch.Tensor): With shape :math:`(C, H, W)`.
        """

        self.relu_outputs = {}
        if guided and not self.relu_registered:
            self._register_relu_hooks()
            self.relu_registered = True

        if torch.cuda.is_available() and use_gpu:
            # self.model = self.model.to('cuda')
            input_ = input_.to('cuda')

        self.model.zero_grad()

        self.gradients = torch.zeros(input_.shape)

        output = self.model(input_)
        target = torch.FloatTensor(1, output.shape[-1]).zero_()

        if torch.cuda.is_available():
            target = target.to('cuda')

        # Set the element at top class index to be 1

        target[0][target_class] = 1 # top_class

        # Calculate gradients of the target class output w.r.t. input_

        output.backward(gradient=target, retain_graph = True)

        # Detach the gradients from the graph and move to cpu

        gradients = self.gradients.detach().cpu()[0]
        
        if take_max:
            # Take the maximum across colour channels
            gradients = gradients.max(dim=0, keepdim=True)[0]

        return gradients

    #####################
    # Private interface #
    #####################

    def _register_conv_hook(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if isinstance(module, nn.modules.conv.Conv2d) and \
                    module.in_channels == 3:
                self.handle_conv.append(module.register_backward_hook(_record_gradients))
                break

    def _register_relu_hooks(self):
        def _record_output(module, input_, output):
            self.relu_outputs[hash(module)] = output

        def _clip_gradients(module, grad_in, grad_out):
            relu_output = self.relu_outputs[hash(module)]
            clippled_grad_out = grad_out[0].clamp(0.0)

            return (clippled_grad_out.mul(relu_output), ) 

        for _, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                self.handle_forward.append(module.register_forward_hook(_record_output))
                self.handle_backward.append(module.register_backward_hook(_clip_gradients))


########################

import os
from torchvision import transforms as torchvision_transforms
from PIL import Image
import PlotNetwork
import matplotlib.pyplot as plt
import numpy as np

class SaliencyMap:
    def __init__(self, dataset, model, experimentName, trial_hash, experiment_params):
        self.dataset = dataset
        self.model = model
        self.experimentName = experimentName
        self.trial_hash = trial_hash
        self.experiment_params = experiment_params

    def getGrayScale(self, img_numpy):
        result = np.dot(img_numpy, [0.299, 0.587, 0.144])
        return result
    
    def display_map_and_predictions(self, heatmap, image_non_normalized, fileName_postfix, fileName, img, patches_mask, layerName, plot=True, use_gpu=False):
        title = fileName.replace('_', '\_')
        if plot:
            fig = plt.figure(figsize=(8, 2.5), dpi= 300)
            
            plt.imshow(self.getGrayScale(format_for_plotting(image_non_normalized).cpu().detach().numpy()),cmap='gray', alpha=0.6) # [:, :, 2] to show a channel
            plt.imshow(self.getGrayScale(format_for_plotting(heatmap).cpu().detach().numpy()), cmap='seismic', alpha=0.5)
            plt.imshow(format_for_plotting(patches_mask).cpu().detach().numpy(), cmap='seismic', alpha=0.3)
            plt.xticks([])
            plt.yticks([])

            fig.tight_layout()
            fig.show()
            path = os.path.join(self.experimentName, "models", self.trial_hash, 'saliency_map', fileName)
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(os.path.join(path,fileName+fileName_postfix+".pdf"), bbox_inches = 'tight',
    pad_inches = 0)
            # fig.suptitle("Saliency Map - " + title)

        if torch.cuda.is_available() and use_gpu:
            img = img.cuda()

        if plot:
            activatins_rows = 1
            A = PlotNetwork.plot_activations(self.model.model, layerName, img, path, self.experiment_params, self.dataset, fileName+fileName_postfix, activatins_rows)
        else:
            activation = PlotNetwork.model_activations(self.model.model, layerName, self.dataset)
            A = activation(img)
        return A

    def getTransformedImage(self, img, augmentation, normalization):
        augmentation2, normalization2, pad2 = self.dataset.toggle_image_loading(augmentation=augmentation, normalization=normalization)
        transforms = self.dataset.getTransforms()
        composedTransforms = torchvision_transforms.Compose(transforms)
        img_clone = composedTransforms(img)
        img_clone = img_clone.unsqueeze(0)
        self.dataset.toggle_image_loading(augmentation2, normalization2, pad2)
        return img_clone

    def getBoundingBox(self, x_indx, y_indx, box_width):
        box_half_width = int(box_width/2)
        x_indx = x_indx-box_half_width
        y_indx = y_indx-box_half_width
        return x_indx, y_indx, box_width, box_width

    def getFiller(self, x_width, y_width, img, green=False):
        dim = img.shape[1]
        detached = img.detach()
        filler = torch.zeros((1, dim,x_width, y_width))
        if not green:
            for i in range(dim):
                filler[0, i, :, :] = detached[0, i, 0, 0]      
        else:
            for i in range(dim):
                filler[0, i, :, :] = torch.ones((1, 1,x_width, y_width)) if i == 1 else torch.zeros((1, 1,x_width, y_width))
        return filler
    
    def getCoordinatedOfHighest(self, tnsor, topk=1):
        rawmaxidx = tnsor.view(-1).topk(topk)[1][topk-1]
        idx = []
        for adim in list(tnsor.size())[::-1]:
            idx.append((rawmaxidx%adim).item())
            rawmaxidx = rawmaxidx // adim
        return idx[:-1]

    def getCoordinatesOfHighestPixel(self, saliency_map, topk=1):
        return self.getCoordinatedOfHighest(saliency_map, topk)
    
    def getCoordinatesOfHighestPatch(self, saliency_map, box_width, topk=1):
        # Do a convolution to get a sum
        filters = torch.ones(1, 1, box_width, box_width)
        # padding = int(torch.floor(torch.tensor([float(box_width)])/2).item())
        padding=0
        stride = box_width
        saliency_map = torch.nn.functional.conv2d(saliency_map.unsqueeze(0), filters, padding=(padding, padding), stride=(stride, stride)).squeeze()
        saliency_map = saliency_map.unsqueeze(0)
        # Get highest pixel after convolution
        return [element * stride + int(stride/2) + 1 for element in self.getCoordinatesOfHighestPixel(saliency_map, topk)] 
        
    def GetSaliencyMap(self, fileName, layerName, maxCovered=False, box_width= None, topLeft=None, topk=1, plot=True, use_gpu=False, generate_all_steps=True):
        title = ntpath.basename(fileName)
        
        isFine = (layerName != 'coarse')
        self.model.setOutputsOfInterest({
            "fine": isFine,
            "coarse" : not isFine
        })
        
        original =  Image.open(fileName)

        image_non_normalized = self.getTransformedImage(original, False, False)
        patches_mask = torch.clone(image_non_normalized)

        image_normalized = self.getTransformedImage(original, False, True)
        image_normalized.requires_grad = True

        if torch.cuda.is_available() and use_gpu:
            image_normalized = image_normalized.cuda()
            
        output = self.model(image_normalized)
        bestClass = torch.max(output, 1)[1]

        backprop = Backprop(self.model)
        saliency_map = backprop.calculate_gradients(image_normalized,
                                         bestClass,
                                         guided=True,
                                         take_max=True, #True
                                         use_gpu=use_gpu)
        title_postfix = ""
        if maxCovered: 
            if topLeft is not None:
                saliency_map_max_x_indx = topLeft[0]
                saliency_map_max_y_indx = topLeft[1]
            else:
                saliency_map_max_y_indx = []
                saliency_map_max_x_indx = []
                # This is not efficient, but OK for now.
                for i in range(topk):
                    saliency_map_max_y_indx_, saliency_map_max_x_indx_ = self.getCoordinatesOfHighestPatch(saliency_map, box_width, i+1)
                    saliency_map_max_x_indx.append(saliency_map_max_x_indx_)
                    saliency_map_max_y_indx.append(saliency_map_max_y_indx_)
            
            title_postfix = " - Occluded - " + str(box_width)
            for i in range(topk):
                saliency_map_max_x_indx_, saliency_map_max_y_indx_, x_width, y_width = self.getBoundingBox(saliency_map_max_x_indx[i], saliency_map_max_y_indx[i], box_width)
                # if plot:
                #     print(saliency_map_max_x_indx_, saliency_map_max_y_indx_)
                filler = self.getFiller(x_width, y_width, image_non_normalized, green=False)
                green_filler = self.getFiller(x_width, y_width, image_non_normalized, green=True)

                image_non_normalized[0, :, saliency_map_max_x_indx_:saliency_map_max_x_indx_ + x_width,
                                        saliency_map_max_y_indx_:saliency_map_max_y_indx_+y_width] = filler
                
                patches_mask[0, :, saliency_map_max_x_indx_:saliency_map_max_x_indx_ + x_width,
                                        saliency_map_max_y_indx_:saliency_map_max_y_indx_+y_width] = green_filler

                image_normalized[0, :, saliency_map_max_x_indx_:saliency_map_max_x_indx_ + x_width,
                                        saliency_map_max_y_indx_:saliency_map_max_y_indx_+y_width] = filler
                if generate_all_steps or (i == topk-1):
                    title_postfix = title_postfix + " - " + str(i+1) 

                    heatmap = visualizeAllClasses(backprop, image_normalized, [bestClass], guided=True, use_gpu=use_gpu)        
                    A = self.display_map_and_predictions(heatmap, image_non_normalized, title_postfix, title, image_normalized, patches_mask, layerName, plot=plot, use_gpu=use_gpu)
        else:
            heatmap = visualizeAllClasses(backprop, image_normalized, [bestClass], guided=True, use_gpu=use_gpu) 
            A = self.display_map_and_predictions(heatmap, image_non_normalized, title_postfix, title, image_normalized, patches_mask, layerName, plot=plot, use_gpu=use_gpu)
        
        # We need this to clear the hooks. del backprop is not working for some reason.
        backprop.__del__() 
        backprop = None

        return saliency_map, A
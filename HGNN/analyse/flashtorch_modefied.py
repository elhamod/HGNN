import warnings

import torch
import torch.nn as nn

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

def visualizeOverlay(self, input_, denormalizedInput_, target_class, guided=False, use_gpu=False, cmap='viridis', alpha=.5):

        # Calculate gradients
        max_gradients = self.calculate_gradients(input_,
                                                 target_class,
                                                 guided=guided,
                                                 take_max=False,
                                                 use_gpu=use_gpu)
        clipped_gradients = format_for_plotting(standardize_and_clip(max_gradients,
                         saturation=1))
        #return two entries of type (image, cmap, alpha)
        return [(format_for_plotting(denormalizedInput_), None, None),
              (clipped_gradients,
               cmap,
               alpha)]
    
def visualizeHeatmap(self, input_, target_class, guided=False, use_gpu=False, cmap='viridis', alpha=.5):

        # Calculate gradients
        max_gradients = self.calculate_gradients(input_,
                                                 target_class,
                                                 guided=guided,
                                                 take_max=True, # We are not taking max because the interplay between channels is important!
                                                 use_gpu=use_gpu)
        #return image, cmap, alpha
        output = (format_for_plotting(standardize_and_clip(max_gradients,
                         saturation=1)),
               cmap,
               alpha)
        return [output,output]
    

def visualizeAllClasses(self, image_normalized, input_non_normalized, listOfClasses, guided=False, use_gpu=False, cmap='viridis', alpha=.5):
    w = input_non_normalized.shape[3]
    h = input_non_normalized.shape[2]
    
    result = torch.zeros(1, 3, h, w*len(listOfClasses))
    idx = 0
    for i in listOfClasses:
        result[:, :, :, idx*w:(idx+1)*w] = self.calculate_gradients(image_normalized,
                                         i,
                                         guided=guided,
                                         take_max=True, #True
                                         use_gpu=use_gpu)
        idx = idx + 1
        
    stdized = standardize_and_clip(result)
    stdized = torch.cat((stdized, input_non_normalized), 3)
    
    output = (format_for_plotting(stdized),
       cmap,
       alpha)
    return output
    
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
        self._register_conv_hook()

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

        if guided:
            self.relu_outputs = []
            self._register_relu_hooks()

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
                module.register_backward_hook(_record_gradients)
                break

    def _register_relu_hooks(self):
        self.relu_outputs = {}
        def _record_output(module, input_, output):
            self.relu_outputs[hash(module)] = output

        def _clip_gradients(module, grad_in, grad_out):
            relu_output = self.relu_outputs[hash(module)]
            clippled_grad_out = grad_out[0].clamp(0.0)

            return (clippled_grad_out.mul(relu_output), ) 

        for _, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(_record_output)
                module.register_backward_hook(_clip_gradients)


########################

import os
from torchvision import transforms as torchvision_transforms
from PIL import Image
import PlotNetwork
import matplotlib.pyplot as plt

class SaliencyMap:
    def __init__(self, dataset, model, experimentName, trial_hash, experiment_params):
        self.dataset = dataset
        self.model = model
        self.experimentName = experimentName
        self.trial_hash = trial_hash
        self.experiment_params = experiment_params
    
    def display_map_and_predictions(self, heatmap, fileName, img, layerName, plot=True, use_gpu=False):
        title = fileName.replace('_', '\_')
        if plot:
            fig = plt.figure(figsize=(8, 2.5), dpi= 300)

            plt.imshow(heatmap.cpu().detach().numpy()) # [:, :, 2] to show a channel
            plt.xticks([])
            plt.yticks([])

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.show()
            path = os.path.join(self.experimentName, "results", self.trial_hash, 'saliency_map')
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(os.path.join(path,fileName+".pdf"), bbox_inches = 'tight',
    pad_inches = 0)
            fig.suptitle("Saliency Map - " + title)

        if torch.cuda.is_available() and use_gpu:
            img = img.cuda()

        if plot:
            activatins_rows = 1
            A = PlotNetwork.plot_activations(self.model.model, layerName, img, path, self.experiment_params, self.dataset, fileName, activatins_rows)
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
        x_indx2 = x_indx+box_half_width
        y_indx2 = y_indx+box_half_width
        x_indx = x_indx-box_half_width if x_indx-box_half_width >=0 else 0
        y_indx = y_indx-box_half_width if y_indx-box_half_width >=0 else 0
        x_width = x_indx2 - x_indx
        y_width = y_indx2 - y_indx
        return x_indx, y_indx, x_width, y_width

    def getFiller(self, x_width, y_width, img):
        detached = img.detach()
        filler = torch.zeros((1, 3,x_width, y_width))
        filler[0, 0, :, :] = detached[0, 0, 0, 0]
        filler[0, 1, :, :] = detached[0, 1, 0, 0]
        filler[0, 2, :, :] = detached[0, 2, 0, 0]
        return filler
    
    def getCoordinatedOfHighest(self, tnsor, topk=1):
        rawmaxidx = tnsor.view(-1).topk(topk)[1][topk-1]
        idx = []
        for adim in list(tnsor.size())[::-1]:
            idx.append((rawmaxidx%adim).item())
            rawmaxidx = rawmaxidx / adim
        return idx[:-1]

    def getCoordinatesOfHighestPixel(self, saliency_map, topk=1):
        return self.getCoordinatedOfHighest(saliency_map, topk)
    
    def getCoordinatesOfHighestPatch(self, saliency_map, box_width, topk=1):
        # Do a convolution to get a sum
        filters = torch.ones(1, 1, box_width, box_width)
        padding = int(torch.floor(torch.tensor([float(box_width)])/2).item())
        stride = box_width
        saliency_map = torch.nn.functional.conv2d(saliency_map.unsqueeze(0), filters, padding=(padding, padding), stride=(stride, stride)).squeeze()
        saliency_map = saliency_map.unsqueeze(0)
        # Get highest pixel after convolution
        return [element * stride for element in self.getCoordinatesOfHighestPixel(saliency_map, topk)] 
        
    def GetSaliencyMap(self, img_full_path, fileName, layerName, maxCovered=False, box_width= None, topLeft=None, topk=1, plot=True, use_gpu=False):
        title = fileName
        
        isFine = (layerName != 'coarse')
        self.model.setOutputsOfInterest({
            "fine": isFine,
            "coarse" : not isFine
        })
        
        original =  Image.open(os.path.join(img_full_path, fileName))

        image_non_normalized = self.getTransformedImage(original, False, False)

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
        if maxCovered:       

            saliency_map_max_x = torch.max(saliency_map, 1)
            saliency_map_max_y = torch.max(saliency_map_max_x[0], 1)
            saliency_map_max_y_indx = saliency_map_max_y[1]
            saliency_map_max_x_indx = saliency_map_max_x[1][0, saliency_map_max_y_indx]

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
            
            title = title + " - Occluded - " + str(box_width)
            for i in range(topk):
                saliency_map_max_x_indx_, saliency_map_max_y_indx_, x_width, y_width = self.getBoundingBox(saliency_map_max_x_indx[i], saliency_map_max_y_indx[i], box_width)
                if plot:
                    print(saliency_map_max_x_indx_, saliency_map_max_y_indx_)

                filler = self.getFiller(x_width, y_width, image_non_normalized)

                image_non_normalized[0, :, saliency_map_max_x_indx_:saliency_map_max_x_indx_ + x_width,
                                        saliency_map_max_y_indx_:saliency_map_max_y_indx_+y_width] = filler

                image_normalized[0, :, saliency_map_max_x_indx_:saliency_map_max_x_indx_ + x_width,
                                        saliency_map_max_y_indx_:saliency_map_max_y_indx_+y_width] = filler

                title_ = title + " - " + str(i+1) 

                heatmap = visualizeAllClasses(backprop, image_normalized, image_non_normalized, [bestClass], guided=True, use_gpu=use_gpu)        
                A = self.display_map_and_predictions(heatmap[0], title_, image_normalized, layerName, plot=plot, use_gpu=use_gpu)
        else:
            heatmap = visualizeAllClasses(backprop, image_normalized, image_non_normalized, [bestClass], guided=True, use_gpu=use_gpu)        
            A = self.display_map_and_predictions(heatmap[0], title, image_normalized, layerName, plot=plot, use_gpu=use_gpu)
        return saliency_map, A
import matplotlib.pylab as plt
import math
import os
import torch
import numpy as np

class model_activations(torch.nn.Module):
    def __init__(self, model, layer_name, useHeirarchy):
        super(model_activations, self).__init__()
        
        if not useHeirarchy:
            self.features = model
        else:
            activations = model.activations
            self.features = (lambda x: activations(x)[layer_name])
        
    def forward(self, x):
        x = self.features(x)
        return x
    
# Define the function for plotting the activations
def plot_activations(model, layer_name, input_img, experimentName, params, title="", n_splits=1):
    activation = model_activations(model, layer_name, params["useHeirarchy"])
    A = activation(input_img)
    if (layer_name == "species" or layer_name == "genus"):
        A = torch.nn.Softmax(dim=1)(A)
    print("Number of activations: ", A.shape)
    
    A = A.squeeze(0).detach().cpu().numpy()
    n_activations = A.shape[0]
    A_min = A.min().item()
    A_max = A.max().item()
    
    A_split = np.array_split(A, n_splits)

    thresh = A.max() / 1.5
    fig, axes = plt.subplots(n_splits, 1, figsize=(25, 1.5*n_splits), dpi= 300)
    idx = 0
    for j, A_single in enumerate(A_split):
        ax = axes[j] if n_splits >1 else axes
        A_single = np.expand_dims(A_single, axis=0)
        feature_num = A_single.shape[1]
        ax.imshow(A_single, vmin=A_min, vmax=A_max, cmap='Blues', extent=[idx-0.5, idx+feature_num-0.5, -0.5, 0.5])
        for i in range(feature_num):
            ax.text(i+idx, 0, "{:0.2f}".format(A_single[0, i]),
                     horizontalalignment="center",
                     color="white" if A[i] > thresh else "black")
        ax.set_xlim(idx-0.5, idx + feature_num-0.5)
        idx = idx + feature_num

   
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(experimentName, title+"_activations.pdf"), bbox_inches = 'tight',
    pad_inches = 0)
    plt.suptitle("Activations - "+title, fontsize=10)
    plt.show()
    return A

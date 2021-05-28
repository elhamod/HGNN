
from torch import nn

# ----Phylogeny criterion
class Phylogeny_MSE():
    def __init__(self, distance_matrix, phylogeny_loss_epsilon):
        self.criterion = nn.MSELoss()
        self.distance_matrix = distance_matrix
        self.epsilon = phylogeny_loss_epsilon

    def __call__(self, pred, true):
        # get d[true, :]
        d = self.distance_matrix[true, :] + self.epsilon

        # get inv
        inv_d = 1/d

        # get MSE
        return self.criterion(inv_d, pred)   

    def cuda(self):
        self.criterion = self.criterion.cuda()
        self.distance_matrix = self.distance_matrix.cuda()
        return self
#------
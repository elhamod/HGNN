import torch

# ----Phylogeny criterion
class Phylogeny_KLDiv():
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.cuda_=False

    def __call__(self, pred, true):
        # print(true[0], pred[0,:])
        # sum_w = torch.sum(self.distance_matrix[true, :], 1).reshape(-1, 1)
        max_w = torch.max(self.distance_matrix[true, :], 1)[0].reshape(-1, 1)
        # w = self.distance_matrix[true, :]/ sum_w
        # w = self.distance_matrix[true, :]/ max_w
        w = max_w - self.distance_matrix[true, :]
        # print('w', w[0,:])

        loss = torch.nn.KLDivLoss()
        loss2 = torch.nn.LogSoftmax(dim=1)
        loss3 = torch.nn.Softmax(dim=1)
        temp = loss2(pred)
        # print('1', loss3(pred)[0, :])
        temp2 = loss3(w)
        # print('1_', temp2[0, :])

        temp = loss(temp, temp2)

        return temp

    def cuda(self):
        self.cuda_ = True
        self.distance_matrix = self.distance_matrix.cuda()
        return self
#------
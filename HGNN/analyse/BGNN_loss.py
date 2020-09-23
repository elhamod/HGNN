import loss_landscapes
import torch


class BGNN_loss(loss_landscapes.Metric):
    
    def __init__(self, loader, lambda_, isDSN):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.isDSN = isDSN
        self.lambda_ = lambda_
        self.loader = loader
        
        self.inputs = []
        self.target = []
        self.meta_target = []
        self.index = -1
        
        
        
    def get_next(self):
        if len(self.inputs) == 0:
            for batch in self.loader:
                self.inputs.append(batch["image"])
                self.target.append(batch["fine"])
                self.meta_target.append(batch["coarse"])
        self.index = self.index+1
        if self.index==len(self.inputs):
            self.index=-1
            return None, None, None
        return self.inputs[self.index], self.target[self.index], self.meta_target[self.index]
    
    
    

    def __call__(self, model_wrapper) -> float:        
        inputs, target, meta_target = self.get_next()
        while inputs is not None:
            if torch.cuda.is_available():
                self.criterion = self.criterion.cuda()
                inputs = inputs.cuda()
                target = target.cuda()
                meta_target = meta_target.cuda()
            
            batch_size = inputs.shape[0]
                
            output = model_wrapper.forward(inputs)
            
            loss_coarse = 0
            if output["coarse"] is not None:
                loss_coarse = self.criterion(output["coarse"], meta_target if not self.isDSN else target)
            loss_fine = self.criterion(output["fine"], target)
            loss = loss_fine + self.lambda_*loss_coarse
            loss = batch_size*loss
            
            inputs, target, meta_target = self.get_next()

        loss = loss/len(self.loader.dataset)
        loss = loss.detach()
        if torch.cuda.is_available():
            loss = loss.cpu()
        
        return loss.numpy()
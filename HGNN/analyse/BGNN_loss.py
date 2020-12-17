import loss_landscapes
import torch


class BGNN_loss_batched(loss_landscapes.Metric):
    
    def __init__(self, loader, lambda_, isDSN, device=None):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.isDSN = isDSN
        self.lambda_ = lambda_
        self.loader = loader
        
        self.inputs = []
        self.target = []
        self.meta_target = []
        self.index = -1
        self.device=device
        
        
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
        with torch.set_grad_enabled(False):
            loss_total = 0
            while inputs is not None:
                if self.device is not None:
                    self.criterion = self.criterion.cuda()
                    inputs = inputs.cuda()
                    target = target.cuda()
                    meta_target = meta_target.cuda()
                else:
                    print("Warning! BGNN_loss_batched on CPU.")
                
                batch_size = inputs.shape[0]
                    
                output = model_wrapper.forward(inputs)
                
                # loss_coarse = 0
                # if output["coarse"] is not None:
                #     loss_coarse = self.criterion(output["coarse"], meta_target if not self.isDSN else target)
                loss_fine = self.criterion(output["fine"], target)
                loss = loss_fine #+ self.lambda_*loss_coarse # We removed loss_coarse because lambda changes over time.
                loss_total = loss_total+batch_size*loss
                
                inputs, target, meta_target = self.get_next()

        loss_total = loss_total/len(self.loader.dataset)
        loss_total = loss_total.detach()
        if self.device is not None:
            loss_total = loss_total.cpu()
        
        return loss_total.numpy()
    
    

    
    
    
    
    
class BGNN_loss_oneBatch(loss_landscapes.Loss):
    def __init__(self, inputs, target, meta_target, lambda_, isDSN, device=None):
        criterion = torch.nn.CrossEntropyLoss()
        if device is not None:
            criterion = criterion.cuda()
            inputs = inputs.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
        super().__init__(criterion, inputs, target)
        self.isDSN = isDSN
        self.lambda_ = lambda_
        self.meta_target = meta_target
        self.device=device

    def __call__(self, model_wrapper) -> float:
        output = model_wrapper.forward(self.inputs)
        
        # loss_coarse = 0
        # if output["coarse"] is not None:
        #     loss_coarse = self.loss_fn(output["coarse"], self.meta_target if not self.isDSN else self.target)
        loss_fine = self.loss_fn(output["fine"], self.target)
        loss = loss_fine #+ self.lambda_*loss_coarse # We removed loss_coarse because lambda changes over time.
        
        loss = loss.detach()
        if self.device is not None:
            loss = loss.cpu()
        
        return loss.numpy()
import torch
from .client import Client
from utils import *

class fedprox(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):   
        super(fedprox, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        
        self.optimizer = self._get_optimizer(self.model.parameters(), optimizer_name=self.args.optimizer, lr=lr, weight_decay=self.args.weight_decay + self.args.lamb)
        
    
    def _get_optimizer(self, parameters, optimizer_name, lr, weight_decay):
        if optimizer_name == 'sgd':
            return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'adam':
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adagrad':
            return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamax':
            return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'nadam':
            return torch.optim.NAdam(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'asgd':
            return torch.optim.ASGD(parameters, lr=lr, weight_decay=weight_decay, t0=1000)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def train(self):
        # local training
        self.model.train()
        
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).reshape(-1).long()
                
                predictions = self.model(inputs)
                loss_pred = self.loss(predictions, labels)
                
                param_list = param_to_vector(self.model)
                delta_list = self.received_vecs['Params_list'].to(self.device)
                loss_correct = torch.sum(param_list * delta_list) * -1.0
                
                loss = loss_pred + self.args.lamb * loss_correct
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm) 
                self.optimizer.step()
                
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs
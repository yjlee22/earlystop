import torch
from .client import Client
from utils import *
from optimizer import *

class fedgamma(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):   
        super(fedgamma, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        
        self.base_optimizer = self._get_optimizer(self.model.parameters(), optimizer_name=self.args.optimizer, lr=lr, weight_decay=self.args.weight_decay)
        self.optimizer = ESAM(self.model.parameters(), self.base_optimizer, rho=self.args.rho)

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
        self.model.train()
        
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).reshape(-1).long()
                
                self.optimizer.paras = [inputs, labels, self.loss, self.model]
                self.optimizer.step()
                
                param_list = param_to_vector(self.model)
                delta_list = self.received_vecs['Local_VR_correction'].to(self.device)
                loss_correct = torch.sum(param_list * delta_list)
                
                loss_correct.backward()
                
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm) 
                self.base_optimizer.step()
                
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list
        
        return self.comm_vecs

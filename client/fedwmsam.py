import torch
from utils import *
from .client import Client
from optimizer import *

def loss_gamma(predictions, labels, param_list, momentum_list, alpha):
    
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    
    return alpha * loss(predictions, labels) + torch.sum(
        param_list * momentum_list) * (1 - alpha)

class fedwmsam(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(fedwmsam, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        
        self.target_model = param_to_vector(self.model).to(self.device)
        self.base_optimizer = self._get_optimizer(self.model.parameters(), optimizer_name=self.args.optimizer, lr=lr, weight_decay=self.args.weight_decay)
        self.sam_optimizer = WMSAM(self.model.parameters(), self.base_optimizer, rho=self.args.rho)
        self.loss = loss_gamma

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
        momentum_list = self.received_vecs['Client_momentum'].to(self.device)
        
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).reshape(-1).long()
                
                # WMSAM Step
                self.sam_optimizer.paras = [inputs, labels, self.loss, self.model, momentum_list, self.args.alpha]
                differ_vector = self.target_model - param_to_vector(self.model).to(self.device)
                self.sam_optimizer.step(differ_vector)

                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                
                # Base Optimizer Step
                self.base_optimizer.step()
                
                # Update target model with momentum
                self.target_model += momentum_list

        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs
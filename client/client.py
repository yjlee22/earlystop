import torch
from utils import *
from dataset import Dataset
from torch.utils import data

class Client():
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        self.args = args
        self.device = device
        self.model_func = model_func
        self.received_vecs = received_vecs
        self.comm_vecs = {
            'local_update_list': None,
            'local_model_param_list': None,
        }

        if self.received_vecs['Params_list'] is None:
            raise Exception("CommError: invalid vectors Params_list received")

        self.model = set_client_from_params(
            device=self.device,
            model=self.model_func(),
            params=self.received_vecs['Params_list']
        )

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = self._get_optimizer(self.model.parameters(), args.optimizer, lr, args.weight_decay)

        self.dataset = data.DataLoader(
            Dataset(dataset[0], dataset[1], train=True, dataset_name=self.args.dataset, args=self.args),
            batch_size=self.args.batchsize,
            shuffle=True,
            pin_memory=True
        )

        self.max_norm = 10

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

        for _ in range(self.args.local_epochs):
            for inputs, labels in self.dataset:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).reshape(-1).long()

                predictions = self.model(inputs)
                loss = self.loss(predictions, labels)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()

        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs

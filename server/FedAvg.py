import torch
from client import *
from .server import Server

class FedAvg(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(FedAvg, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
        }
        self.Client = fedavg
        
    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                    * (self.server_model_params_list - self.clients_params_list[client]))
        
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        return self.server_model_params_list + Averaged_update

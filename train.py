import torch
import timm
import argparse
import random

from utils import *
from server import *
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BloodMNIST')          
parser.add_argument('--model', type=str, default='convnextv2_nano')
parser.add_argument('--non-iid', action='store_true', default=False)
parser.add_argument('--pretrain', action='store_true', default=False)                                                                              
parser.add_argument('--split-rule',type=str, default='Dirichlet')  
parser.add_argument('--split-coef', default=0.1, type=float)                                                                                                                                    
parser.add_argument('--active-ratio', default=0.1, type=float)                                        
parser.add_argument('--total-client', default=100, type=int)                                         
parser.add_argument('--comm-rounds', default=500, type=int)                                               
parser.add_argument('--num_class', default=8, type=int)                                              
parser.add_argument('--local-epochs', default=5, type=int)                                  
parser.add_argument('--batchsize', default=256, type=int)                                                   
parser.add_argument('--weight-decay', default=0.001, type=float)                                          
parser.add_argument('--local-learning-rate', default=0.001, type=float)
parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad', 'adamax', 'nadam', 'asgd'],)                                                                                      
parser.add_argument('--lr-decay', default=0.998, type=float)                                               
parser.add_argument('--seed', default=0, type=int)                                                       
parser.add_argument('--data-file', default='./', type=str)                                                
parser.add_argument('--out-file', default='out/', type=str)                                             
parser.add_argument('--save-model', action='store_true', default=False)                                  
parser.add_argument('--use-RI', action='store_true', default=False)                                      
parser.add_argument('--alpha', default=0.001, type=float)                                                    
parser.add_argument('--beta', default=0.1, type=float)                                                     
parser.add_argument('--beta1', default=0.9, type=float)                                                   
parser.add_argument('--beta2', default=0.99, type=float)                                                 
parser.add_argument('--lamb', default=0.001, type=float)                                                  
parser.add_argument('--rho', default=0.001, type=float)                                                     
parser.add_argument('--gamma', default=1.0, type=float)                                                  
parser.add_argument('--epsilon', default=0.01, type=float)
parser.add_argument('--gpu', type=int, default=0)                                                
parser.add_argument('--method', choices=['FedAvg', 'FedCM', 'FedDyn', 'SCAFFOLD', 'FedAdam', 'FedProx', 'FedSAM', 'MoFedSAM', \
                                         'FedGamma', 'FedSpeed', 'FedSMOO', 'FedLESAM', 'FedWMSAM'], type=str, default='FedAvg')

parser.add_argument('--threshold', type=float, default=0.005, help='Threshold for early stoppiing')
parser.add_argument('--patience', type=int, default=10, help='Number of rounds to wait before stopping if stagnation persists')
parser.add_argument('--fast', action='store_true', default=False)                                  
parser.add_argument('--validation', action='store_true', help='Run validation-based early stopping only')
parser.add_argument('--proposed', action='store_true', help='Run proposed early stopping only')
                                         
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

args.out_file = f'{args.dataset.lower()}/seed_{args.seed}'

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

if __name__=='__main__':
    if not args.non_iid:
        data_obj = DatasetObject(n_client=args.total_client, seed=args.seed, rule='iid', data_path=args.data_file, args=args)
        print("Initialize the Dataset     --->  {:s} {:s} {:d} clients".format(args.dataset, 'IID', args.total_client))
    else:
        data_obj = DatasetObject(n_client=args.total_client, seed=args.seed, rule=args.split_rule, rule_arg=args.split_coef, data_path=args.data_file, args=args)
        print("Initialize the Dataset     --->  {:s} {:s}-{:s} {:d} clients".format(args.dataset, args.split_rule, str(args.split_coef), args.total_client))

    model_func = lambda: timm.create_model(args.model, pretrained=args.pretrain, in_chans=3, num_classes=args.num_class)
        
    print("Initialize the Model Func  --->  {:s} model".format(args.model))
    init_model = model_func()
    total_trainable_params = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
    print("                           --->  {:d} parameters".format(total_trainable_params))
    
    init_par_list = get_mdl_params(init_model)
    
    server_func = None
    if args.method == 'FedAvg':
        server_func = FedAvg
    elif args.method == 'FedCM':
        server_func = FedCM
    elif args.method == 'FedDyn':
        server_func = FedDyn
    elif args.method == 'SCAFFOLD':
        server_func = SCAFFOLD
    elif args.method == 'FedAdam':
        server_func = FedAdam
    elif args.method == 'FedProx':
        server_func = FedProx
    elif args.method == 'FedSAM':
        server_func = FedSAM
    elif args.method == 'MoFedSAM':
        server_func = MoFedSAM
    elif args.method == 'FedGamma':
        server_func = FedGamma
    elif args.method == 'FedSpeed':
        server_func = FedSpeed
    elif args.method == 'FedSMOO':
        server_func = FedSMOO
    elif args.method == 'FedLESAM':
        server_func = FedLESAM
    elif args.method == 'FedWMSAM':
        server_func = FedWMSAM
    else:
        raise NotImplementedError('not implemented method yet')
    
    _server = server_func(device=device, model_func=model_func, init_model=init_model, init_par_list=init_par_list,
                          datasets=data_obj, method=args.method, args=args)
    _server.train()
    
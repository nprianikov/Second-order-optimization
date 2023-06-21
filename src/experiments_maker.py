from typing import Tuple, Dict

import torch.utils.data
from torch import nn
import torchmetrics
import numpy as np
from src.optimizers.s_bfgs import LBFGS, FullBatchLBFGS
from src.optimizers.k_bfgs import K_BFGS

import src.data_setup as data_setup
import src.optimizers.hessianfree as hessianfree
from src.model_builder import ModelCNN
import utils.config_manager as cm

def make(config: Dict, device: torch.device, **kwargs) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader,
                                                      torch.utils.data.DataLoader, torch.optim.Optimizer, torchmetrics.Accuracy]:
    """
    Makes the experiment, i.e. loads the model, the data, the optimizer and the criterion
    Arguments:
        config: dictionary containing the configuration of the experiment
        device: torch.device object
        **kwargs: additional arguments for the specific optimizer
    """
    # set the seed for reproducibility
    np.random.seed(cm.random_seed)
    torch.manual_seed(cm.random_seed)
    torch.cuda.manual_seed(cm.random_seed)
    
    # choose model
    activation_fn_dict = {"tanh": nn.Tanh, "relu":nn.ReLU, 'sigmoid': nn.Sigmoid}
    model = ModelCNN(
        model_name=config["model"],
        input_shape=3 if config["dataset"] == "cifar10" else 1,
        activation_fn=activation_fn_dict[config["activation_fn"].lower()], 
        p=config["dropout"],
        dataset=config["dataset"]).to(device)
   
    
    # load data
    train_data_loader, test_data_loader = data_setup.train_test_loaders(dataset=config['dataset'],
                                                                        batch_size=config['batch_size'], 
                                                                        slice_size=config["slice_size"])
    
    # choose criterion
    n_classes = 10 # all datasets considered have 10 classes
    criterion = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes, average='macro')
    criterion.to(device)
    
    # choose optimizer
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "HessianFree":
        optimizer = hessianfree.HessianFree(params=model.parameters(), eps=1e-3, **kwargs)
    elif config["optimizer"] == "S_BFGS(L)":
        optimizer = LBFGS(model.parameters(), lr=1., history_size=50, line_search='Wolfe', debug=False)
    elif config["optimizer"] == "K_BFGS":
        optimizer = K_BFGS(model, train_data_loader, algorithm='K-BFGS', lr=config["lr"], lambda_damping=1, debug=False, verbose=False)
    elif config["optimizer"] == "K_BFGS(L)":
        optimizer = K_BFGS(model, train_data_loader, algorithm='K-BFGS(L)', lr=config["lr"], lambda_damping=1, debug=True, verbose=False)
    else:
        raise ValueError("Unknown optimizer type")

    return model, train_data_loader, test_data_loader, optimizer, criterion

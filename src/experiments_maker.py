from typing import Tuple, Dict

import torch.utils.data
from torch import nn
import torchmetrics

import data_setup
import optimizers.hessianfree as hessianfree
from model_builder import SmallCNN, DepthCNN, WidthCNN, DepthWidthCNN


def make(config: Dict, device: torch.device, **kwargs) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader,
                                                      torch.utils.data.DataLoader, torch.optim.Optimizer, torchmetrics.Accuracy]:
    """
    Makes the experiment, i.e. loads the model, the data, the optimizer and the criterion
    Arguments:
        config: dictionary containing the configuration of the experiment
        device: torch.device object
        **kwargs: additional arguments for the specific optimizer
    """
    # choose model
    activation_fn_dict = {"tanh": nn.Tanh, "relu":nn.ReLU, 'sigmoid': nn.Sigmoid}
    models_dict = {"SmallCNN": SmallCNN, "DepthCNN": DepthCNN, "WidthCNN": WidthCNN, "DepthWidthCNN": DepthWidthCNN}
    model = models_dict[config["model"]](
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
    
    # choose optimizer
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "HessianFree":
        optimizer = hessianfree.HessianFree(params=model.parameters(), **kwargs)
    elif config["optimizer"] == "PB_BFGS":
        # TODO: add PB_BFGS
        pass
    elif config["optimizer"] == "K_BFGS":
        # TODO: add K_BFGS
        pass
    elif config["optimizer"] == "K_LBFGS":
        # TODO: add K_LBFGS
        pass
    elif config["optimizer"] == "K_FAC":
        #TODO add K_FAC
        pass
    else:
        raise ValueError("Unknown optimizer type")

    return model, train_data_loader, test_data_loader, optimizer, criterion

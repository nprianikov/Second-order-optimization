from typing import Tuple, Dict

import torch.utils.data
import torchmetrics

import data_setup
import optimizers.hessianfree
from model_builder import SmallCNN, DepthCNN, WidthCNN, DepthWidthCNN


def make(config: Dict, device: torch.device) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader,
                                                      torch.utils.data.DataLoader, torch.optim.Optimizer, torchmetrics.Accuracy]:
    if config is None:
        config = {
            "epochs": 3,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "dataset": "mnist",
            "optimizer": "SGD",
            "model": "SmallCNN",
            "architecture": "CNN"
        }

    # choose model
    if config["model"] == "SmallCNN":
        model = SmallCNN(dataset=config["dataset"]).to(device)
    elif config["model"] == "DepthCNN":
        model = DepthCNN(dataset=config["dataset"]).to(device)
    elif config["model"] == "WidthCNN":
        model = WidthCNN(dataset=config["dataset"]).to(device)
    elif config["model"] == "DepthWidthCNN":
        model = DepthWidthCNN(dataset=config["dataset"]).to(device)
    else:
        raise ValueError("Unknown model type")
    
    # load data
    train_data_loader, test_data_loader = data_setup.train_test_loaders(dataset=config['dataset'],
                                                                        batch_size=config['batch_size'])
    
    # choose criterion
    criterion = torchmetrics.Accuracy(task='multiclass', num_classes=len(train_data_loader.dataset.classes), average='macro')
    
    # choose optimizer
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "HessianFree":
        optimizer = optimizers.hessianfree.HessianFree(params=model.parameters())
    elif config["optimizer"] == "PB_BFGS":
        # TODO: add PB_BFGS
        pass
    elif config["optimizer"] == "K_BFGS":
        # TODO: add K_BFGS
        pass
    elif config["optimizer"] == "K_LBFGS":
        # TODO: add K_LBFGS
        pass
    else:
        raise ValueError("Unknown optimizer type")

    return model, train_data_loader, test_data_loader, optimizer, criterion

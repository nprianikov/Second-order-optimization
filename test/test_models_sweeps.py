import sys
sys.path.append("../Second-order-optimization")

import torch
from torch import nn
import torchmetrics
import numpy as np
import src.engine as engine
import src.experiments_maker as experiments_maker
import wandb

# set the seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# experiment parameters
wandb_modes = ["disabled", "online"]
wandb_logs = ["all", "gradients", "parameters", None]
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()

datasets_names = ['mnist', 'tmnist','fashion_mnist', 'cifar10']
optimizers_names = ['SGD', 'HessianFree', 'PB_BFGS', 'K_BFGS', 'K_LBFGS']
models_names = ['SmallCNN', 'DepthCNN', 'WidthCNN', 'DepthWidthCNN']


def main():
    wandb.init(project="baselines-grid-sweeps")
    # make the model, data and optimization
    main_config = wandb.config
    model, train_dataloader, test_dataloader, optimizer, criterion = experiments_maker.make(main_config, device)
    engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, criterion, device, main_config)

sweep_config = {
    'method': 'grid',
    'metric': {
        'goal': 'minimize',
        'name': 'train_loss'
    },
    'parameters': {
        'epochs': {'values': [5]},
        'learning_rate': {'values': [0.001]},
        'batch_size': {'values': [32]},
        'dataset': {'values': datasets_names},
        'optimizer': {'values': [optimizers_names[0]]},
        'model': {'values': models_names},
        'architecture': {'values': ["CNN"]},
        'wandb_log': {'values': [wandb_logs[1]]},
        'wandb_log_freq': {'values': [1000]},
        'wandb_log_batch': {'values': [1000]},
        'slice_size': {'values': [1.0] },
        'activation_fn': {'values': ["Tanh", "ReLU"]},
        'dropout': {'values': [0.0]},
        'checkpoints': {'values': [0]},
    }
}

sweep_id = wandb.sweep(
    sweep_config, 
    project="baselines-grid-sweeps"
    )

wandb.agent(sweep_id, function=main)

wandb.finish()

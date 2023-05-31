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

def main():
    wandb.init(project="baselines-sweeps")
    # make the model, data and optimization
    main_config = wandb.config
    print(main_config)
    model, train_dataloader, test_dataloader, optimizer, criterion = experiments_maker.make(
        main_config, device, 
        damping=main_config.damping, supress_extremes=main_config.supress_extremes, cg_max_iter=main_config.cg_max_iter)
    engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, criterion, device, main_config)

sweep_config = {
    'method': 'random',
    'metric': {
        'goal': 'minimize',
        'name': 'train_loss'
    },
    'parameters': {
        'damping': {'min': 0.0, 'max': 1e-2},
        'supress_extremes': {'min': 0.65, 'max': 0.95},
        'cg_max_iter': {'values': [100, 200, 300, 400, 500]}
    }
}

config = {
    'epochs': 1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'dataset':  'cifar10',
    'optimizer': 'HessianFree',
    'model': 'DepthWidthCNN',
    'architecture': "CNN",
    'wandb_log': wandb_logs[0],
    'wandb_log_freq': 1,
    'wandb_log_batch': 1,
    'slice_size': 0.5,
    'activation_fn': "Tanh",
    'dropout': 0.0,
    'checkpoints': 0,
}

def wrap_values(config):
    return {key: {'value': value} for key, value in config.items()}

sweep_config['parameters'].update(wrap_values(config))

sweep_id = wandb.sweep(
    sweep_config, 
    project="baselines-sweeps"
    )

wandb.agent(sweep_id, function=main, count=10)

wandb.finish()

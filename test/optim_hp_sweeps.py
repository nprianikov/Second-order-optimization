import sys
sys.path.append("..")

import torch
from torch import nn
import torchmetrics
import numpy as np
import src.engine as engine
import src.experiments_maker as experiments_maker
import wandb
import utils.config_manager as cm

# Create an argument parser
parser = cm.create_parser()
# Parse the command-line arguments
args = parser.parse_args()

def main():
    wandb.init(project="optimizer-hyperparams")
    # make the model, data and optimization
    main_config = wandb.config
    print(main_config)
    model, train_dataloader, test_dataloader, optimizer, criterion = experiments_maker.make(
        main_config, cm.device, 
        damping=main_config.damping, supress_extremes=main_config.supress_extremes, cg_max_iter=main_config.cg_max_iter)
    engine.train(model, train_dataloader, test_dataloader, cm.loss_fn, optimizer, criterion, cm.device, main_config)


sweep_config = {
    'method': 'random',
    'metric': {
        'goal': 'minimize',
        'name': 'train_loss'
    },
    'parameters': {
        'damping': {'min': 0.5, 'max': 0.95},
        'supress_extremes': {'min': 0.65, 'max': 0.95},
        'cg_max_iter': {'values': [100, 200, 300, 400, 500]}
    }
}

config = cm.create_config(vars(args))

def wrap_values(config):
    return {key: {'value': value} for key, value in config.items()}

sweep_config['parameters'].update(wrap_values(config))

sweep_id = wandb.sweep(
    sweep_config, 
    project="optimizer-hyperparams"
    )

wandb.agent(sweep_id, function=main, count=10)

wandb.finish()

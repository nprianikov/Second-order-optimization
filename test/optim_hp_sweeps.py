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
        # damping=main_config.damping, 
        # supress_extremes=main_config.supress_extremes, 
        lr=main_config.alpha,
        delta_decay=main_config.delta_decay,
        cg_max_iter=main_config.cg_max_iter
    )
    engine.train(model, train_dataloader, test_dataloader, cm.loss_fn, optimizer, criterion, cm.device, main_config)


sweep_config = {
    'method': 'random',
    'metric': {
        'goal': 'minimize',
        'name': 'train_loss'
    },
    'parameters': {
        'delta_decay': {'values': [0.85, 0.90, 0.95, 0.99]},
        'cg_max_iter': {'values': [25, 50, 75, 100]},
        'alpha': {'values': [0.001, 0.01, 0.1, 1.0]}
    }
}

# 'parameters': {
#         'damping': {'values':{[1e-2, 1e-1, 0.25, 0.50]}},           - used for preconditioner
#         'supress_extremes': {'values':{[0.65, 0.75, 0.85, 0.95]}},  - used for preconditioner
#         'delta_decay': {'values': {[0.85, 0.90, 0.95, 0.99]}},
#         'cg_max_iter': {'values': {[25, 50, 75, 100]}},
#         'lr': {'values': [0.1, 0.25, 0.5, 1.0]}
#     }

config = cm.create_config(vars(args),
                          epochs=5,
                          batch_size=128,
                          dataset=cm.datasets_names[0],
                          optimizer=cm.optimizers_names[1],
                          model=cm.models_names[0],
                          wandb_log_batch=50)

def wrap_values(config):
    return {key: {'value': value} for key, value in config.items()}

sweep_config['parameters'].update(wrap_values(config))

sweep_id = wandb.sweep(
    sweep_config, 
    project="optimizer-hyperparams"
    )

wandb.agent(sweep_id, function=main, count=10)

wandb.finish()

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

if args.loop == 1:

    def main():
        wandb.init(project="experiments-grid-sweeps")
        main_config = wandb.config.__dict__['_items']
        model, train_dataloader, test_dataloader, optimizer, criterion = experiments_maker.make(main_config, cm.device)
        engine.train(model, train_dataloader, test_dataloader, cm.loss_fn, optimizer, criterion, cm.device, main_config)
    # sweep config
    sweep_config = {'method': 'grid', 'metric': {'goal': 'minimize', 'name': 'train_loss'}, 'parameters': { }}
    # specify parameters: lists for iterables
    config = cm.create_config(vars(args),
                              epochs=5,
                              dataset=cm.datasets_names,
                              optimizer=['SGD', 'S_BFGS(L)'], #cm.optimizers_names[2],
                              model=cm.models_names,
                              wandb_log_freq=0,
                              wandb_log_batch=32,
                              batch_size=128,
                              checkpoints=10)
    # update sweep config
    sweep_config['parameters'].update(cm.wrap_values(config))
    # init sweep
    sweep_id = wandb.sweep(sweep_config, project="experiments-grid-sweeps")
    wandb.agent(sweep_id, function=main)

else:

    config = cm.create_config(vars(args))

    with wandb.init(project="experiments-grid-sweeps", config=config, mode=cm.wandb_modes[args.wandb_mode]):
        model, train_dataloader, test_dataloader, optimizer, criterion = experiments_maker.make(config, cm.device)
        engine.train(model, train_dataloader, test_dataloader, cm.loss_fn, optimizer, criterion, cm.device, config)

wandb.finish()

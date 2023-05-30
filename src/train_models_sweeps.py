import torch
from torch import nn
import torchmetrics
import numpy as np
import src.engine as engine
import src.experiments_maker as experiments_maker
import wandb
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Train a model')

# Add the command-line arguments
parser.add_argument('--loop', default=1, type=int, required=True, help='Loop over all the combinations of the datasets, optimizers and models. 0: Disabled, 1: Enabled')
parser.add_argument('--epochs', default=1, type=int, required=False, help='Number of epochs to train for')
parser.add_argument('--lr', default=1e-3, type=float, required=False, help='Learning rate for training')
parser.add_argument('--batch_size', default=32, type=int, required=False, help='Batch size for training')
parser.add_argument('--dataset', default="mnist", type=str, required=False, help='Name of the dataset to train on: mnist, tmnist, fashion_mnist, cifar10')
parser.add_argument('--optimizer', default="SGD", type=str, required=False, help='Name of the optimizer to train: SGD, HessianFree, PB_BFGS, K_BFGS, K_LBFGS')
parser.add_argument('--model', default="SmallCNN", type=str, required=False, help='Name of the model to train: SmallCNN, DepthCNN, WidthCNN, DepthWidthCNN')
parser.add_argument('--wandb_mode', default=1, type=int, required=False, help='Wandb mode. 0: Disabled, 1: Online')
parser.add_argument('--wandb_log', default=0, type=int, required=False, help='Wandb log extra information. 0: All, 1: Gradients, 2: Parameters, 3: None')
parser.add_argument('--wandb_log_freq', default=1, type=int, required=False, help='Wandb log frequency of extra information.')
parser.add_argument('--wandb_log_batch', default=0, type=int, required=False, help='Wandb log frequency of metrics per training batch. 0 for disabled.')
parser.add_argument('--data_slice', default=1.0, type=float, required=False, help='Specify the slice of the dataset to use, e.g. 0.5: Half, 1: All')
parser.add_argument('--activation_fn', default=0, type=str, required=False, help='Activation function to use. 0: Tanh, 1: ReLU, 2: Sigmoid')
parser.add_argument('--dropout', default=0.0, type=float, required=False, help='Dropout rate to use. 0.0: Disabled, 0.5: 50%')


# Parse the command-line arguments
args = parser.parse_args()

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
        'damping': {'min': 0.5, 'max': 0.95},
        'supress_extremes': {'min': 0.65, 'max': 0.95},
        'cg_max_iter': {'values': [100, 200, 300, 400, 500]}
    }
}

config = dict(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        dataset=args.dataset, 
        optimizer=args.optimizer, 
        model=args.model,
        architecture="CNN",
        wandb_log=wandb_logs[args.wandb_log],
        wandb_log_freq=args.wandb_log_freq,
        slice_size=args.data_slice,
        activation_fn=args.activation_fn,
        dropout=args.dropout,
    )

def wrap_values(config):
    return {key: {'value': value} for key, value in config.items()}

sweep_config['parameters'].update(wrap_values(config))

sweep_id = wandb.sweep(
    sweep_config, 
    project="baselines-sweeps"
    )

wandb.agent(sweep_id, function=main, count=10)

wandb.finish()

import argparse
import torch
from torch import nn


ARG_DEFS = {
    'loop': 0,
    'epochs': 1,
    'lr': 1e-3,
    'batch_size': 32,
    'dataset': "mnist",
    'optimizer': "SGD",
    'model': "SmallCNN",
    'wandb_mode': 0,
    'wandb_log': 3,
    'wandb_log_freq': 0,
    'wandb_log_batch': 0,
    'slice_size': 1.0,
    'activation_fn': 'Tanh',
    'dropout': 0.0,
    'checkpoints': 0
}

wandb_modes = ["disabled", "online"]
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()
random_seeds = [42, 47, 68, 27, ]

datasets_names = ['mnist', 'tmnist','fashion_mnist', 'cifar10']
optimizers_names = ['SGD', 'HessianFree', 'PB_BFGS', 'K_BFGS', 'K_LBFGS']
models_names = ['SmallCNN', 'DepthCNN', 'WidthCNN', 'DepthWidthCNN']


def create_parser():
    '''
    Create a parser for the command-line arguments with the default values:\n
    loop = 0\n
    epochs = 1\n 
    lr = 0.001\n 
    batch_size = 32\n 
    dataset = mnist\n 
    optimizer = SGD\n 
    model = SmallCNN\n
    wandb_mode = 0\n 
    wandb_log = 3\n 
    wandb_log_freq = 0\n 
    wandb_log_batch = 0\n 
    slice_size = 1.0\n 
    activation_fn = Tanh\n 
    dropout = 0.0\n 
    checkpoints = 0\n
    '''
    # parser
    parser = argparse.ArgumentParser(description='Train a model')

    # Add the command-line arguments
    parser.add_argument('--loop', default=ARG_DEFS['loop'], type=int, required=False, help='Loop over all the combinations of the datasets, optimizers and models. 0: Disabled, 1: Enabled')
    parser.add_argument('--epochs', default=ARG_DEFS['epochs'], type=int, required=False, help='Number of epochs to train for')
    parser.add_argument('--lr', default=ARG_DEFS['lr'], type=float, required=False, help='Learning rate for training')
    parser.add_argument('--batch_size', default=ARG_DEFS['batch_size'], type=int, required=False, help='Batch size for training')
    parser.add_argument('--dataset', default=ARG_DEFS['dataset'], type=str, required=False, help='Name of the dataset to train on: mnist, tmnist, fashion_mnist, cifar10')
    parser.add_argument('--optimizer', default=ARG_DEFS['optimizer'], type=str, required=False, help='Name of the optimizer to train: SGD, HessianFree, PB_BFGS, K_BFGS, K_LBFGS')
    parser.add_argument('--model', default=ARG_DEFS['model'], type=str, required=False, help='Name of the model to train: SmallCNN, DepthCNN, WidthCNN, DepthWidthCNN')
    parser.add_argument('--wandb_mode', default=ARG_DEFS['wandb_mode'], type=int, required=False, help='Wandb mode. 0: Disabled, 1: Online')
    parser.add_argument('--wandb_log', default=ARG_DEFS['wandb_log'], type=int, required=False, help='Wandb log extra information. 0: All, 1: Gradients, 2: Parameters, 3: None')
    parser.add_argument('--wandb_log_freq', default=ARG_DEFS['wandb_log_freq'], type=int, required=False, help='Wandb log frequency of extra information.')
    parser.add_argument('--wandb_log_batch', default=ARG_DEFS['wandb_log_batch'], type=int, required=False, help='Wandb log frequency of metrics per training batch. 0 for disabled.')
    parser.add_argument('--slice_size', default=ARG_DEFS['slice_size'], type=float, required=False, help='Specify the slice of the dataset to use, e.g. 0.5: Half, 1: All')
    parser.add_argument('--activation_fn', default=ARG_DEFS['activation_fn'], type=str, required=False, help='Activation function to use. 0: Tanh, 1: ReLU, 2: Sigmoid')
    parser.add_argument('--dropout', default=ARG_DEFS['dropout'], type=float, required=False, help='Dropout rate to use. 0.0: Disabled, 0.5: 50%')
    parser.add_argument('--checkpoints', default=ARG_DEFS['checkpoints'], type=int, required=False, help='Number of model checkpoints to save per epoch.')

    return parser


def wrap_values(config):
    return {key: {'values': value} if isinstance(value, list) else {'value': value} for key, value in config.items()}


def config_to_desc(config):
    config_str = ""
    for key, value in config.items():
        config_str += f"{key} = {value}\n "
    return config_str


def create_config(*args, **kwargs):
    '''
    Create a config dictionary from the command line arguments and typed arguments.
    Deafult values are used if no arguments are provided:\n
    epochs = 1\n 
    lr = 0.001\n 
    batch_size = 32\n 
    dataset = mnist\n 
    optimizer = SGD\n 
    model = SmallCNN\n 
    wandb_log = 3\n 
    wandb_log_freq = 0\n 
    wandb_log_batch = 0\n 
    slice_size = 1.0\n 
    activation_fn = Tanh\n 
    dropout = 0.0\n 
    checkpoints = 0\n
    '''
    config = ARG_DEFS.copy()
    # update whole config with args
    if len(args) > 0:
        config.update(args[0])
    # update some entries with kwargs
    if len(kwargs) > 0:
        config.update(kwargs)

    return config



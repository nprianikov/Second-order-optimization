import torch
from torch import nn
import torchmetrics
import numpy as np
import engine
import experiments_maker
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

if args.loop == 1:

    # datasets_names = ['mnist', 'tmnist','fashion_mnist', 'cifar10']
    # optimizers_names = ['SGD', 'HessianFree', 'PB_BFGS', 'K_BFGS', 'K_LBFGS']
    # models_names = ['SmallCNN', 'DepthCNN', 'WidthCNN', 'DepthWidthCNN']

    # experimental run v0.2
    datasets_names = ['tmnist']
    optimizers_names = ['SGD']#, 'HessianFree']
    models_names = ['SmallCNN']

    for dataset_name in datasets_names:
        for optimizer_name in optimizers_names:
            for model_name in models_names:
                config = dict(
                    epochs=args.epochs,
                    learning_rate=args.lr,
                    batch_size=args.batch_size,
                    dataset=dataset_name, # iterable
                    optimizer=optimizer_name, # iterable
                    model=model_name, # iterable
                    architecture="CNN",
                    wandb_log=wandb_logs[args.wandb_log],
                    wandb_log_freq=args.wandb_log_freq,
                )

                with wandb.init(project="baselines_cnn", config=config, mode=wandb_modes[args.wandb_mode]):
                    config = wandb.config
                    # make the model, data and optimization
                    model, train_dataloader, test_dataloader, optimizer, criterion = experiments_maker.make(config, device)
                    engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, criterion, device, config)

                wandb.finish()
else:

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
    )

    with wandb.init(project="baselines_cnn", config=config, mode=wandb_modes[args.wandb_mode]):
        config = wandb.config
        # make the model, data and optimization
        model, train_dataloader, test_dataloader, optimizer, criterion = experiments_maker.make(config, device)
        engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, criterion, device, config)

    wandb.finish()

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

config = {
    'epochs': 1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'dataset':  datasets_names[0],
    'optimizer': optimizers_names[0],
    'model': models_names[0],
    'architecture': "CNN",
    'wandb_log': wandb_logs[3],
    'wandb_log_freq': 0,
    'wandb_log_batch': 0,
    'slice_size': 1.0,
    'activation_fn': "Tanh",
    'dropout': 0.0,
    'checkpoints': 10,
}

with wandb.init(project="baselines_cnn", config=config, mode=wandb_modes[0]):
    # make the model, data and optimization
    model, train_dataloader, test_dataloader, optimizer, criterion = experiments_maker.make(
        config, device)
    engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, criterion, device, config)

wandb.finish()

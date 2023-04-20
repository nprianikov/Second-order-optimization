import torch
from torch import nn
import torchmetrics
import numpy as np
import engine
import experiments_maker
import wandb

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

modes = ["disabled", "online"]
device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 50
lr = 1e-3
classes = 10
loss_fn = nn.CrossEntropyLoss()
accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=classes, average='macro')

batch_size=16

# datasets_names = ['mnist', 'tmnist','fashion_mnist', 'cifar10']
# optimizers_names = ['SGD', 'HessianFree', 'PB_BFGS', 'K_BFGS', 'K_LBFGS']
# models_names = ['SmallCNN', 'DepthCNN', 'WidthCNN', 'DepthWidthCNN']

# experimental run v0.1
datasets_names = ['tmnist']
optimizers_names = ['SGD']#, 'HessianFree']
models_names = ['SmallCNN']

for dataset_name in datasets_names:
    for optimizer_name in optimizers_names:
        for model_name in models_names:
            config = dict(
                epochs=epochs,
                classes=classes,
                learning_rate=lr,
                batch_size=batch_size,
                dataset=dataset_name, # iterable
                optimizer=optimizer_name, # iterable
                model=model_name, # iterable
                architecture="CNN"
            )

            with wandb.init(project="baselines_cnn", config=config, mode=modes[0]):
                config = wandb.config
                # make the model, data and optimization
                model, train_dataloader, test_dataloader, optimizer = experiments_maker.make(config, device)
                engine.train(config['epochs'], model, train_dataloader, test_dataloader, loss_fn, optimizer, accuracy_fn, device)

            wandb.finish()

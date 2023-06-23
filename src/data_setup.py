import sys
import os
from typing import Tuple, Any, Union

import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor



def train_test_loaders(dataset: str, batch_size=32, slice_size=1.0):
    """
    Returns dataloaders for the specified dataset name
    :param dataset: name of the dataset
    :param batch_size: batch size. 0 for full batch
    :return: train and test dataloaders
    """
    root_path = os.path.join(os.getcwd(), '..', 'datasets')
    # load full datasets
    if dataset == "mnist":
        train_data = datasets.MNIST(root=root_path, train=True, download=True, transform=ToTensor(),
                                    target_transform=None)
        test_data = datasets.MNIST(root=root_path, train=False, download=True, transform=ToTensor())
    elif dataset == "fashion_mnist":
        train_data = datasets.FashionMNIST(root=root_path, train=True, download=True, transform=ToTensor(),
                                           target_transform=None)
        test_data = datasets.FashionMNIST(root=root_path, train=False, download=True, transform=ToTensor())
    elif dataset == "cifar10":
        train_data = datasets.CIFAR10(root=root_path, train=True, download=True, transform=ToTensor(),
                                      target_transform=None)
        test_data = datasets.CIFAR10(root=root_path, train=False, download=True, transform=ToTensor())
    else:
        return ValueError("Dataset not supported")
    
    # slice the dataset
    if slice_size < 1.0:
        train_data = balanced_slice(train_data, 10, slice_size)
        test_data = balanced_slice(test_data, 10, slice_size)

    # create dataloaders
    train_dataloader = DataLoader(train_data, batch_size=(len(train_data) if batch_size == 0 else batch_size), shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=(len(test_data) if batch_size == 0 else batch_size), shuffle=False)   
    
    return train_dataloader, test_dataloader


def balanced_slice(full_dataset, n_classes, slice_size):
    # ids of class samples
    class_subset = {}
    for i in range(n_classes):
        class_subset[i] = [idx for idx, (_, label) in enumerate(full_dataset) if label == i]
    # sizes
    subset_size = int(len(full_dataset) * slice_size)
    class_subset_size = subset_size // n_classes
    # getting sample slices per class
    balanced_subset_indices = []
    for i in range(n_classes):
        subset_indices = torch.randperm(len(class_subset[i]))[:class_subset_size].tolist()
        balanced_subset_indices.extend(subset_indices)
    # taking the subset
    balanced_subset = Subset(full_dataset, balanced_subset_indices)

    return balanced_subset

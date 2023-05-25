from typing import Tuple, Any, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class TMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.classes = ['0 - zero',
                        '1 - one',
                        '2 - two',
                        '3 - three',
                        '4 - four',
                        '5 - five',
                        '6 - six',
                        '7 - seven',
                        '8 - eight',
                        '9 - nine']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data.drop(columns=["names", "labels"]).iloc[idx].values.reshape(28, 28)
        label = self.data["labels"].iloc[idx]
        if self.transform:
            image = self.transform(image)
            image = image.type(torch.float32)
        return image, label


def train_test_loaders(dataset: str, batch_size=32):
    """
    Returns dataloaders for the specified dataset name
    :param dataset: name of the dataset
    :param batch_size: batch size
    :return: train and test dataloaders
    """
    if dataset == "mnist":
        train_data = datasets.MNIST(root="datasets", train=True, download=True, transform=ToTensor(),
                                    target_transform=None)
        test_data = datasets.MNIST(root="datasets", train=False, download=True, transform=ToTensor())
    elif dataset == "fashion_mnist":
        train_data = datasets.FashionMNIST(root="datasets", train=True, download=True, transform=ToTensor(),
                                           target_transform=None)
        test_data = datasets.FashionMNIST(root="datasets", train=False, download=True, transform=ToTensor())
    elif dataset == "cifar10":
        train_data = datasets.CIFAR10(root="datasets", train=True, download=True, transform=ToTensor(),
                                      target_transform=None)
        test_data = datasets.CIFAR10(root="datasets", train=False, download=True, transform=ToTensor())
    elif dataset == "tmnist":
        tmnist_dataset = TMNISTDataset(csv_file="../datasets/TMNIST/TMNIST_Data.csv", transform=ToTensor())
        # split the dataset into train and test with a fixed random seed and ratio 5:1
        train_size = int(0.8 * len(tmnist_dataset))
        test_size = len(tmnist_dataset) - train_size
        train_data, test_data = torch.utils.data.random_split(tmnist_dataset, [train_size, test_size])
    else:
        return ValueError("Dataset not supported")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

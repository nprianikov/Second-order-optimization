from timeit import default_timer as timer
from typing import Tuple

import wandb
import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader, # type: ignore
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: torchmetrics.Metric,
               device: torch.device) -> Tuple[float, float]:
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)
        # Optimizer step
        if optimizer.__class__.__name__ == "HessianFree":
            optimizer.zero_grad()
            _y_pred = model(X)
            _loss = loss_fn(_y_pred, y)

            # _loss.backward()

            def closure():
                return _loss, _y_pred

            loss, y_pred = optimizer.step(closure=closure) # type: ignore
        else:
            # Forward pass
            y_pred = model(X)
            # Calculate loss
            loss = loss_fn(y_pred, y)
            # Optimizer zero grad
            optimizer.zero_grad()
            # Loss backward
            loss.backward()
            optimizer.step()

        train_loss += loss
        train_acc += accuracy_fn(y_pred.argmax(dim=1), y).item()
    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def test_step(data_loader: torch.utils.data.DataLoader, # type: ignore
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn: torchmetrics.Metric,
              device: torch.device) -> Tuple[float, float]:
    test_loss, test_acc = 0, 0
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred = model(X)
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(test_pred.argmax(dim=1), y).item()
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    return test_loss, test_acc


def train(epochs: int,
          model: torch.nn.Module,
          train_data_loader: torch.utils.data.DataLoader, # type: ignore
          test_data_loader: torch.utils.data.DataLoader, # type: ignore
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy_fn: torchmetrics.Metric,
          device: torch.device
          ):
    wandb.watch(model, loss_fn, log="all")

    for epoch in tqdm(range(epochs)):
        # train loop
        train_time_start = timer()
        train_loss, train_acc = train_step(data_loader=train_data_loader,
                                           model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_fn=accuracy_fn,
                                           device=device)
        train_time_end = timer()
        total_train_time_model = train_time_end - train_time_start
        # test loop
        test_time_start = timer()
        test_loss, test_acc = test_step(data_loader=test_data_loader,
                                        model=model,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device)
        test_time_end = timer()
        total_test_time_model = train_time_end - train_time_start
        # log metrics to wandb
        wandb.log({"model": model.__class__.__name__, "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                   "total_train_time": total_train_time_model, "test_loss": test_loss, "test_acc": test_acc,
                   "total_test_time": total_test_time_model})

        # save model
        torch.onnx.export(model, next(iter(train_data_loader))[0], f"{model.__class__.__name__}.onnx")
        # artifact = wandb.use_artifact("int_pb/hf-end-to-end-deployment/model-1lv0fpmm:v0")  # TODO: change to your artifact
        onnx_artifact = wandb.Artifact(f"{model.__class__.__name__}-onnx", type="model")
        onnx_artifact.add_file(f"{model.__class__.__name__}.onnx")

        wandb.log_artifact(onnx_artifact)

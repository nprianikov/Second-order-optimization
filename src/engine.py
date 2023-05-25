from timeit import default_timer as timer
from typing import Tuple

import wandb
import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import datetime
from optimizers.hessianfree import empirical_fisher_diagonal_batched
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader, # type: ignore
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: torchmetrics.Metric,
               device: torch.device) -> Tuple[float, float]:
    train_loss, train_acc = 0, 0
    batch_counter = 0
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)
        # Optimizer step
        if optimizer.__class__.__name__ == "HessianFree":
            # closure
            def closure():
                _y_pred = model(X)
                _loss = loss_fn(_y_pred, y)
                _loss.backward(create_graph=True)
                return _loss, _y_pred
            
            def M_inv():  # inverse preconditioner
                return empirical_fisher_diagonal_batched(model, X, y, loss_fn)

            optimizer.zero_grad()
            loss, y_pred = optimizer.step(closure=closure, M_inv=M_inv) # type: ignore

            # clear gradients due to create_graph=True
            # for param in model.parameters():
            #     param.grad = None

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

        if batch_counter % 10 == 0:
            print(f"Batch: {batch_counter}\nLoss: {loss.item()}\n-------")

        train_loss += loss.item()
        train_acc += accuracy_fn(y_pred.argmax(dim=1), y).item()
        batch_counter = batch
    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= batch_counter
    train_acc /= batch_counter
    return train_loss, train_acc


def test_step(data_loader: torch.utils.data.DataLoader, # type: ignore
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn: torchmetrics.Metric,
              device: torch.device) -> Tuple[float, float]:
    test_loss, test_acc = 0, 0
    model.eval()  # put model in eval mode
    batch_counter = 0
    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred = model(X)
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y).item()
            test_acc += accuracy_fn(test_pred.argmax(dim=1), y).item()
            batch_counter = batch
        # Adjust metrics and print out
        test_loss /= batch_counter
        test_acc /= batch_counter
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_data_loader: torch.utils.data.DataLoader, # type: ignore
          test_data_loader: torch.utils.data.DataLoader, # type: ignore
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy_fn: torchmetrics.Metric,
          device: torch.device,
          config: dict,
          ):

    now = datetime.datetime.now()
    print("-------")
    print(f"New experiment started at {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Config: {config}\n")
    print("-------")

    wandb.watch(model, loss_fn, log=config["wandb_log"], log_freq=config["wandb_log_freq"])

    for epoch in tqdm(range(config["epochs"])):

        print(f"Epoch: {epoch}\n-------")

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
        total_test_time_model = test_time_end - test_time_start

        print(f"Train_loss: {train_loss:.5f} | Train_acc: {train_acc:.2f} | Total_train_time: {total_train_time_model} | \
              Test_loss: {test_loss:.5f} | Test_acc: {test_acc:.2f} | Total_test_time: {total_test_time_model}\n")

        # log metrics to wandb
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                   "total_train_time": total_train_time_model, "test_loss": test_loss, "test_acc": test_acc,
                   "total_test_time": total_test_time_model})

    # save model
    # torch.onnx.export(model, next(iter(train_data_loader))[0], f"{model.__class__.__name__}.onnx")
    # onnx_artifact = wandb.Artifact(f"{model.__class__.__name__}-onnx", type="model")
    # onnx_artifact.add_file(f"{model.__class__.__name__}.onnx")
    # wandb.log_artifact(onnx_artifact)

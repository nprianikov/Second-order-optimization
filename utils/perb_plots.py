import os
import sys
sys.path.append("..")

import copy
import json
import numpy as np
import torch 
from torchvision import datasets, transforms
from pyhessian import hessian # Hessian computation
#from utils.pyhessian import hessian # Hessian computation
from utils.density_plot import get_esd_plot # ESD computation
import src.experiments_maker as experiments_maker
import src.engine as engine
import utils.config_manager as cm
from pyhessian.utils import normalization

import matplotlib.pyplot as plt


def load_checkpoint_model(f, **kwargs):
    # read the dict from txt file
    config_path = os.path.join(f, '..', 'config.txt')
    with open(config_path) as file:
        data = file.read()
    config = json.loads(data)

    model, train_dataloader, test_dataloader, optimizer, accuracy_fn = experiments_maker.make(config, cm.device, **kwargs)
    model.load_state_dict(torch.load(f))
    model.eval()

    # create a copy of the model
    model_perb = copy.deepcopy(model)
    model_perb.load_state_dict(torch.load(f))
    model_perb.eval()

    return model, model_perb, train_dataloader, test_dataloader, optimizer, accuracy_fn, config


def continue_training(f, **kwargs):
    model, _, train_dataloader, test_dataloader, optimizer, accuracy_fn, config = load_checkpoint_model(f, **kwargs)
    engine.train(model, train_dataloader, test_dataloader, cm.loss_fn, optimizer, accuracy_fn, cm.device, config)


def perb_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb


def perb_params_2d(model_orig, model_perb, directions, alphas):
    for m_orig, m_perb, d1, d2 in zip(model_orig.parameters(), model_perb.parameters(), directions[0], directions[1]):
        m_perb.data = m_orig.data + alphas[0] * d1 + alphas[1] * d2
    return model_perb


def top_eig_perb_plot(f, radius=5e-1, n=21, **kwargs):
    
    model, model_perb, train_dataloader, _, _, _, _ = load_checkpoint_model(f, **kwargs)

    # get the loss landscape
    lams = np.linspace(-radius, radius, n).astype(np.float32)
    loss_list = []

    for inputs, targets in train_dataloader:
        break

    # create the hessian computation module
    hessian_comp = hessian(model, cm.loss_fn, data=(inputs, targets), cuda=False)
    _, top_eigenvector = hessian_comp.eigenvalues(maxIter=100, tol=1e-2, top_n=2)

    for lam in lams:
        model_perb = perb_params(model, model_perb, top_eigenvector[0], lam)
        loss_list.append(cm.loss_fn(model_perb(inputs), targets).item())

    # plot y horizontal line for point with min loss
    plt.axhline(y=min(loss_list), color='r', linestyle='--')
    plt.plot(lams, loss_list)
    # display min loss on y axis
    ytickValues = plt.yticks()[0]
    new_ytick_value = min(loss_list)
    ytick_values = np.append(ytickValues, new_ytick_value)
    ytick_labels = [str(round(val, 2)) for val in ytick_values]
    plt.yticks(ytick_values, ytick_labels)

    plt.ylabel('Loss')
    plt.xlabel('Perturbation')
    plt.title('Loss landscape perturbed based on top Hessian eigenvector')

    plt.show()


def top_2_eig_perb_plot(f, radius=5e-1, n=21, **kwargs):
    
    model, model_perb, train_dataloader, _, _, _, _ = load_checkpoint_model(f, **kwargs)

    for inputs, targets in train_dataloader:
        break

    # create the hessian computation module
    hessian_comp = hessian(model, cm.loss_fn, data=(inputs, targets), cuda=False)
    _, top_eigenvector = hessian_comp.eigenvalues(maxIter=100, tol=1e-2, top_n=2)

    # Define the perturbation directions and alphas
    directions = [top_eigenvector[0], top_eigenvector[1]]  # Provide your own second direction

    alphas = np.meshgrid(np.linspace(-radius, radius, n).astype(np.float32),
                        np.linspace(-radius, radius, n).astype(np.float32))

    loss_list = []

    # Perturb the model and calculate loss for each combination of alphas
    for i in range(len(alphas[0])):
        for j in range(len(alphas[1])):
            model_perb = perb_params_2d(model, model_perb, directions=directions, alphas=[alphas[0][i][j], alphas[1][i][j]])
            loss_list.append(cm.loss_fn(model_perb(inputs), targets).item())


    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(alphas[1], alphas[0], np.array(loss_list).reshape(alphas[0].shape), cmap='coolwarm', edgecolor='none')
    # smaller font
    plt.rcParams.update({'font.size': 8})
    ax.set_xlabel('Alpha 2')
    ax.set_ylabel('Alpha 1')
    ax.set_zlabel('Loss')

    ax.view_init(elev=35, azim=45)
    #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.6, 1]))

    # Show the plot
    plt.show()


def rand_perb_plot(f, radius=5e-1, n=21, **kwargs):

    model, model_perb, train_dataloader, _, _, _, _ = load_checkpoint_model(f, **kwargs)

    for inputs, targets in train_dataloader:
        break

    v = [torch.randn_like(p) for p in model.parameters()]
    v = normalization(v)

    lams = np.linspace(-radius, radius, n).astype(np.float32)
    loss_list = []

    for lam in lams: 
        model_perb = perb_params(model, model_perb, v, lam)
        loss_list.append(cm.loss_fn(model_perb(inputs), targets).item())

    plt.plot(lams, loss_list)
    plt.ylabel('Loss')
    plt.xlabel('Perturbation')
    plt.title('Loss landscape perturbed based on a random direction')


def rand_2d_perb_plot(f, radius=5e-1, n=21, **kwargs):

    model, model_perb, train_dataloader, _, _, _, _ = load_checkpoint_model(f, **kwargs)

    for inputs, targets in train_dataloader:
        break

    v1 = normalization([torch.randn_like(p) for p in model.parameters()])
    v2 = normalization([torch.randn_like(p) for p in model.parameters()])

    # Define the perturbation directions and alphas
    directions = [v1, v2]  # Provide your own second direction
    alphas = np.meshgrid(np.linspace(-radius, radius, n).astype(np.float32),
                        np.linspace(-radius, radius, n).astype(np.float32))
    loss_list = []
    # Perturb the model and calculate loss for each combination of alphas
    for i in range(len(alphas[0])):
        for j in range(len(alphas[1])):
            model_perb = perb_params_2d(model, model_perb, directions=directions, alphas=[alphas[0][i][j], alphas[1][i][j]])
            loss_list.append(cm.loss_fn(model_perb(inputs), targets).item())


    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(alphas[1], alphas[0], np.array(loss_list).reshape(alphas[0].shape), cmap='coolwarm', edgecolor='none')
    # smaller font
    plt.rcParams.update({'font.size': 8})
    ax.set_xlabel('Alpha 2')
    ax.set_ylabel('Alpha 1')
    ax.set_zlabel('Loss')

    ax.view_init(elev=35, azim=45)
    #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.6, 1]))

    # Show the plot
    plt.show()


def grad_perb_plot(f, radius=5e-1, n=21, **kwargs):
    
    model, model_perb, train_dataloader, _, _, _, _ = load_checkpoint_model(f, **kwargs)

    # get the loss landscape
    lams = np.linspace(-radius, radius, n).astype(np.float32)
    loss_list = []

    for inputs, targets in train_dataloader:
        break

    # generate gradient vector to do the loss plot
    loss = cm.loss_fn(model_perb(inputs), targets)
    loss.backward()

    v = [p.grad.data for p in model_perb.parameters()]
    v = normalization(v)
    model_perb.zero_grad()

    for lam in lams: 
        model_perb = perb_params(model, model_perb, v, lam)
        loss_list.append(cm.loss_fn(model_perb(inputs), targets).item())

    plt.plot(lams, loss_list)
    plt.ylabel('Loss')
    plt.xlabel('Perturbation')
    plt.title('Loss landscape perturbed based on gradient direction')

    plt.show()


def hessian_esd_plot(f, **kwargs):
    model, _, train_dataloader, _, _, _, _ = load_checkpoint_model(f, **kwargs)

    for inputs, targets in train_dataloader:
        break

    hessian_comp = hessian(model, cm.loss_fn, data=(inputs, targets), cuda=False)
    
    trace = hessian_comp.trace()
    print("The trace of this model is: %.4f"%(np.mean(trace)))

    density_eigen, density_weight = hessian_comp.density()
    get_esd_plot(density_eigen, density_weight)


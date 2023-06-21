import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy
import copy
import utils.config_manager as cm


def get_zero_torch(params):
    layers_params = params['layers_params']
    device = params['device']
    
    delta = []
    for l in range(len(layers_params)):
        delta_l = {}
        delta_l['W'] = torch.zeros(layers_params[l]['output_size'], layers_params[l]['input_size'], device=device)
        delta_l['b'] = torch.zeros(layers_params[l]['output_size'], device=device)
        delta.append(delta_l)
        
    return delta


def get_plus_torch(model_grad, delta):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = model_grad[l][key].to(cm.device) + delta[l][key].to(cm.device)
        sum_p.append(sum_p_l)
    return sum_p


def get_if_nan(p):
    for l in range(len(p)):
        for key in p[l]:
            if torch.sum(p[l][key] != p[l][key]):
                return True
    return False


def get_multiply_scalar(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha * delta[l][key]
        alpha_p.append(alpha_p_l)
    return alpha_p


def get_multiply_scalar_no_grad(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha * delta[l][key].data.to(cm.device)
        alpha_p.append(alpha_p_l)
    return alpha_p


def get_opposite(delta):
    numlayers = len(delta)
    p = []
    for l in range(numlayers):
        p_l = {}
        for key in delta[l]:
            p_l[key] = -delta[l][key]
        p.append(p_l)
        
    return p


def get_model_grad(model, params):
    device = params['device']
    model_grad_torch = []
    grouped = zip(*[iter(model.parameters())]*2)
    for l, (param1, param2) in enumerate(grouped):
        model_grad_torch_l = {}
        if model.layers_params[l]['name'] == 'conv':
            model_grad_torch_l['W'] = copy.deepcopy(param1.grad).reshape(param1.data.size()[0], -1).to(device)
        elif model.layers_params[l]['name'] == 'fc':
            model_grad_torch_l['W'] = copy.deepcopy(param1.grad).to(device)
        model_grad_torch_l['b'] = copy.deepcopy(param2.grad).to(device)
        model_grad_torch.append(model_grad_torch_l)
    return model_grad_torch


def get_homo_grad(model_grad_N1, params):
    device = params['device']

    homo_model_grad_N1 = []
    for l in range(params['numlayers']):
        homo_model_grad_N1_l = torch.cat((model_grad_N1[l]['W'], model_grad_N1[l]['b'].unsqueeze(1)), dim=1)
        homo_model_grad_N1.append(homo_model_grad_N1_l)

    return homo_model_grad_N1  
    
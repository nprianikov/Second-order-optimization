import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from optimizers.k_bfgs_utils import *
import utils.config_manager as cm


class K_BFGS(torch.optim.Optimizer):
    """
    Implements the Kronecker-factored LBFGS algorithm presented in 
    `Kronecker-factored Quasi-Newton Methods for Deep Learning` https://doi.org/10.48550/arXiv.2102.06737.
    
    Arguments:
        model (torch.nn.Module): model to optimize.
        train_dataset (torch.utils.data.DataLoader): training dataset.
        algorithm (str, optional): algorithm to use. Can be 'K-BFGS' or 'K-BFGS(L)'. Default: 'K-BFGS'
        lr (float, optional): learning rate. Default: 0.3
        lambda_damping (float, optional): damping for (s, y). Default: 0.3
        rho_momentum (float, optional): gradient momentum. Default: 0.9
        tau_weight_decay (float, optional): weight decay. Default: 1e-5
        beta_A_decay (float, optional): decay rate for A inverse. Default: 0.9
        t_inv_freq (int, optional): frequency of the inverse update. Default: 20
        history_size (int, optional): number of (s, y) to store. Default: 50
        debug (bool, optional): if True, uses 1 mini-batch for A inverse approximation. Default: False
        verbose (bool, optional): if True, prints information about the training. Default: True
    
    Adapted from: https://github.com/renyiryry/kbfgs_neurips2020_public
    """

    def __init__(self, model,
                 train_dataset, 
                algorithm = 'K-BFGS', # ['K-BFGS', 'K-BFGS(L)']
                lr = 0.3,
                lambda_damping = 0.3,
                rho_momentum = 0.9,
                tau_weight_decay = 1e-5,
                beta_A_decay = 0.9,
                t_inv_freq = 20,
                history_size = 50,
                debug = False,
                verbose = True):

        if not (algorithm in ['K-BFGS', 'K-BFGS(L)']):
            raise ValueError("Invalid algorithm: {}".format(algorithm))

        defaults = dict(model=model,
                        train_dataset=train_dataset,
                        algorithm=algorithm,
                        lr=lr,
                        lambda_damping=lambda_damping,
                        rho_momentum=rho_momentum,
                        tau_weight_decay=tau_weight_decay,
                        beta_A_decay=beta_A_decay,
                        t_inv_freq=t_inv_freq,
                        history_size=history_size,
                        debug=debug,
                        verbose=verbose)
        super(K_BFGS, self).__init__(model.parameters(), defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "K_BFGS doesn't support per-parameter options (parameter groups)")

        params = {}
        torch.cuda.empty_cache()

        params['debug'] = debug
        params['alpha'] = lr
        params['tau'] = tau_weight_decay

        params['algorithm'] = algorithm
   
        params['Kron_BFGS_A_decay'] = beta_A_decay
        params['Kron_LBFGS_Hg_initial'] = 1
        params['Kron_BFGS_action_h'] = 'Hessian-action-BFGS' 
        params['Kron_BFGS_A_LM_epsilon'] = np.sqrt(lambda_damping)
        params['Kron_BFGS_H_epsilon'] = np.sqrt(lambda_damping)
        params['Kron_BFGS_A_inv_freq'] = t_inv_freq

        params['Kron_BFGS_if_homo'] = True 
        
        if algorithm == 'K-BFGS':
            params['Kron_BFGS_H_initial'] = 1 # B           
            params['Kron_BFGS_action_a'] = 'BFGS' # B 
        
        if algorithm == 'K-BFGS(L)':
            params['Kron_BFGS_action_a'] = 'LBFGS' # L
            params['Kron_BFGS_number_s_y'] = history_size # L
            
        params['seed_number'] = cm.random_seed

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = cm.device 
        params['device'] = device
        params['momentum_gradient_rho'] = rho_momentum
        

        params['N1'] = train_dataset.batch_size
        params['N2'] = params['N1']
        params['num_train_data'] = len(train_dataset)
        params['layers_params'] = model.layers_params 
        params['numlayers'] = model.numlayers 
        
        data_ = {}
        data_['model'] = model
        data_['dataset'] = train_dataset
        data_, params = train_initialization(data_, params)
        data_['model_grad_momentum'] = get_zero_torch(params)

        self.data_ = data_
        self.params = params


    def step(self, data_, params):
        """
            Performs a single optimization step.
        """

        ### step 1: update model weights using current Ha and Hg per layer
        numlayers = params['numlayers']
        data_['model_homo_grad_used_torch'] = get_homo_grad(data_['model_grad_used_torch'], params)

        delta = []
        for l in range(numlayers):
            action_h = params['Kron_BFGS_action_h']
            action_a = params['Kron_BFGS_action_a']
            step_ = 1
            delta_l, data_ = Kron_BFGS_update_per_layer(data_, params, l, action_h, action_a, step_)
            delta.append(delta_l)
            
        p = get_opposite(delta)
        data_['p_torch'] = p
        

        ### step 2: update inverse matrices Ha and Hg per layer ###

        # intermediate weights update for a copy of the model
        # model_new = copy.deepcopy(data_['model'])
        model_new = type(data_['model'])()
        model_new.load_state_dict(data_['model'].state_dict())

        for l in range(model_new.numlayers):
            for key in model_new.layers_weights[l]:
                model_new.layers_weights[l][key].data += params['alpha'] * p[l][key].data

        # forward-backward pass
        z_next = model_new.forward(data_['X_mb'])
        a_next = model_new.a
        h_next = model_new.h
        loss = cm.loss_fn(z_next, data_['t_mb'])

        model_new.zero_grad()
        loss.backward()
        
        a_grad_next = [len(data_['X_mb']) * (a_l.grad) for a_l in a_next]
        for l in range(params['numlayers']):
            if params['layers_params'][l]['name'] == 'conv':
                a_grad_next[l] = a_grad_next[l].mean(dim=[2,3])
        data_['a_grad_next'] = a_grad_next

        a_next, h_next = reshape_a_h(a_next, h_next, params['layers_params'])
        data_['h_next'] = h_next
        data_['a_next'] = a_next
        
        for l in range(numlayers):
            action_h = params['Kron_BFGS_action_h']
            action_a = params['Kron_BFGS_action_a']
            step_ = 2
            _, data_ = Kron_BFGS_update_per_layer(data_, params, l, action_h, action_a, step_)
            
        return data_, params


def Kron_BFGS_update_per_layer(data_, params, l, action_h, action_a, step_):
    i = params['i']
    algorithm = params['algorithm']
    N1 = params['N1']
    
    if step_ == 1:
        device = params['device']
        model_homo_grad = data_['model_homo_grad_used_torch']
        Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]
            
        if i == 0:
            Kron_BFGS_matrices_l['H'] = {}
            if action_h in ['Hessian-action-BFGS']: 
                Kron_BFGS_matrices_l['H']['h'] = torch.eye(data_['h_N2'][l].size()[1], device=device)
            
            if action_a == 'BFGS':
                Kron_BFGS_matrices_l['H']['a_grad'] = torch.eye(
                data_['a_grad_N2'][l].size()[1], device=device, requires_grad=False)
                
                Kron_BFGS_matrices_l['H']['a_grad'] *= params['Kron_LBFGS_Hg_initial']

        if action_h in ['Hessian-action-BFGS', 'Hessian-action-LBFGS']:
            # update A
            A_l = Kron_BFGS_matrices_l['A']
            if params['N1'] < params['num_train_data']*params['N1'] and i == 0:
                1
            else:
                beta_ = params['Kron_BFGS_A_decay']
                homo_h_l = torch.cat((data_['h_N2'][l], torch.ones(N1, 1, device=device)), dim=1)
                
                decay_ = beta_
                weight_ = 1-beta_
            
                A_l = decay_ * A_l + weight_ * torch.mm(homo_h_l.t(), homo_h_l).data / data_['h_N2'][l].size()[0]

            Kron_BFGS_matrices_l['A'] = A_l

            if action_h in ['Hessian-action-BFGS', 'Hessian-action-LBFGS']:
                epsilon_ = params['Kron_BFGS_A_LM_epsilon'] * np.sqrt(data_['model'].layers_params[l]['tau'])

                A_l_LM = Kron_BFGS_matrices_l['A'] + epsilon_ * torch.eye(Kron_BFGS_matrices_l['A'].size(0), device=device)
                Kron_BFGS_matrices_l['A_LM'] = A_l_LM
                
                if action_h in ['Hessian-action-BFGS']:
                    if i == 0:
                        Kron_BFGS_matrices_l['H']['h'] = A_l_LM.inverse()
                    
        data_['Kron_BFGS_matrices'][l] = Kron_BFGS_matrices_l
        delta_l = Kron_BFGS_compute_direction(model_homo_grad, l, data_, params)
        return delta_l, data_
    
    elif step_ == 2:
        if i % params['Kron_BFGS_A_inv_freq'] != 0:
            return [], data_

        Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]
        
        a_grad_next = data_['a_grad_next']
        a_next = data_['a_next']
    
        if action_a == 'BFGS':
            H_l_a_grad = Kron_BFGS_matrices_l['H']['a_grad']
        
        # compute s
        s_l_a = torch.mean(data_['a_N2'][l], dim=0).data - torch.mean(a_next[l], dim=0).data

        # compute y
        mean_a_grad_l = torch.mean(data_['a_grad_N2'][l], dim=0)
        mean_a_grad_next_l = torch.mean(a_grad_next[l], dim=0)
        y_l_a = mean_a_grad_l - mean_a_grad_next_l
             

        if N1 < params['num_train_data']:
            decay_ = 0.9
        else:
            decay_ = 0

        data_['Kron_BFGS_momentum_s_y'][l]['s'] =\
        decay_ * data_['Kron_BFGS_momentum_s_y'][l]['s'] + (1-decay_) * s_l_a

        data_['Kron_BFGS_momentum_s_y'][l]['y'] =\
        decay_ * data_['Kron_BFGS_momentum_s_y'][l]['y'] + (1-decay_) * y_l_a

        s_l_a = data_['Kron_BFGS_momentum_s_y'][l]['s']
        y_l_a = data_['Kron_BFGS_momentum_s_y'][l]['y']
                
        s_l_a, y_l_a = kron_bfgs_update_damping(s_l_a, y_l_a, l, data_, params)
    
        if params['Kron_BFGS_action_a'] == 'LBFGS':
            
            data_['Kron_LBFGS_s_y_pairs']['a'][l] =\
            Kron_LBFGS_append_s_y(
                s_l_a, 
                y_l_a, 
                data_['Kron_LBFGS_s_y_pairs']['a'][l],
                mean_a_grad_l,
                params['Kron_LBFGS_Hg_initial'],
                params
            )
            
        elif params['Kron_BFGS_action_a'] == 'BFGS':
            Kron_BFGS_matrices_l['H']['a_grad'], update_status =\
        get_BFGS_formula(H_l_a_grad, s_l_a, y_l_a, mean_a_grad_l)
        

        if action_h in ['Hessian-action-BFGS', 'Hessian-action-LBFGS']:
            mean_h_l = torch.mean(data_['h_N2'][l], dim=0).data
            
#             mean_h_l = torch.cat((mean_h_l, torch.mean(mean_h_l).unsqueeze(0)), dim=0)
            mean_h_l = torch.cat(
                                (mean_h_l, torch.ones(1, device=params['device'])),
                                dim=0
                            )
        
            if action_h == 'Hessian-action-LBFGS':
                s_l_h = LBFGS_Hv(
                    mean_h_l,
                    data_['Kron_LBFGS_s_y_pairs']['h'][l],
                    params
                )
            elif action_h in ['Hessian-action-BFGS']:
                
                H_l_h = Kron_BFGS_matrices_l['H']['h']
                s_l_h = torch.mv(H_l_h, mean_h_l)
                

            y_l_h = torch.mv(Kron_BFGS_matrices_l['A_LM'], s_l_h)
            if action_h == 'Hessian-action-LBFGS':
                data_['Kron_LBFGS_s_y_pairs']['h'][l] =\
                Kron_LBFGS_append_s_y(
                    s_l_h,
                    y_l_h,
                    data_['Kron_LBFGS_s_y_pairs']['h'][l],
                    [],
                    params['Kron_LBFGS_Ha_initial'],
                    params
                )
            elif action_h in ['Hessian-action-BFGS']:
                Kron_BFGS_matrices_l['H']['h'], update_status =\
                get_BFGS_formula(H_l_h, s_l_h, y_l_h, mean_h_l)
                
        elif action_h == 'BFGS':
            h_next = data_['h_next']
            
            H_l_h = Kron_BFGS_matrices_l['H']['h']
        
            
            # compute s
            mean_h_l = torch.mean(data_['h_N2'][l], dim=0).data
            
            s_l_h = torch.mv(H_l_h, mean_h_l)
            s_l_h = s_l_h * np.sqrt(params['alpha'])

            # compute y
            mean_h_next_l = torch.mean(h_next[l], dim=0).data
            y_l_h = mean_h_l - mean_h_next_l
            
            
            Kron_BFGS_matrices_l['H']['h'] = get_BFGS_formula(H_l_h,
                                                   s_l_h, y_l_h,
                                                      mean_h_l)
        
        data_['Kron_BFGS_matrices'][l] = Kron_BFGS_matrices_l
        
        return [], data_


def get_BFGS_PowellHDamping(s_l_a, y_l_a, alpha, l, data_, params):
    Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]

    if params['Kron_BFGS_action_a'] == 'LBFGS':
        1
    elif params['Kron_BFGS_action_a'] == 'BFGS':
        H_l_a_grad = Kron_BFGS_matrices_l['H']['a_grad']

    s_T_y = torch.dot(s_l_a, y_l_a)
    if params['Kron_BFGS_action_a'] == 'LBFGS':
        Hy = LBFGS_Hv(
            y_l_a,
            data_['Kron_LBFGS_s_y_pairs']['a'][l],
            params
        )
    elif params['Kron_BFGS_action_a'] == 'BFGS':
        Hy = torch.mv(H_l_a_grad ,y_l_a)


    yHy = torch.dot(y_l_a, Hy)
    sy_over_yHy_before = s_T_y.item() / yHy.item()

    if sy_over_yHy_before > alpha:
        theta = 1
        damping_status = 0
    else:
        theta =  ((1-alpha) * yHy / (yHy - s_T_y)).item()
        original_s_l_a = s_l_a
        s_l_a = theta * s_l_a + (1-theta) * Hy
        damping_status = 1
    
    return s_l_a, y_l_a, sy_over_yHy_before


def kron_bfgs_update_damping(s_l_a, y_l_a, l, data_, params):
    s_l_a, y_l_a, _ = get_BFGS_PowellHDamping(s_l_a, y_l_a, 0.2, l, data_, params)
    s_l_a, y_l_a = get_BFGS_ModifiedDamping(s_l_a, y_l_a, l, data_, params)
        
    return s_l_a, y_l_a


def get_BFGS_ModifiedDamping(s_l_a, y_l_a, l, data_, params):
    alpha = params['Kron_BFGS_H_epsilon'] * 1 / np.sqrt(params['layers_params'][l]['tau'])

    s_T_s = torch.dot(s_l_a, s_l_a)
    s_T_y = torch.dot(s_l_a, y_l_a)

    if s_T_y / s_T_s > alpha:
        damping_status = 0
    else:
        theta =  (1-alpha) * s_T_s / (s_T_s - s_T_y)
        y_l_a = theta * y_l_a + (1-theta) * s_l_a
        damping_status = 1
    
    return s_l_a, y_l_a


def Kron_BFGS_compute_direction(model_homo_grad, l, data_, params):
    delta_l = {}
    
    if params['Kron_BFGS_action_a'] == 'LBFGS':
        delta_l_W = LBFGS_Hv(
                model_homo_grad[l],
                data_['Kron_LBFGS_s_y_pairs']['a'][l],
                params
            )

    elif params['Kron_BFGS_action_a'] == 'BFGS':
        Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]
        H_l_a_grad = Kron_BFGS_matrices_l['H']['a_grad']
        delta_l_W = torch.mm(H_l_a_grad, model_homo_grad[l])
    
    
    if params['Kron_BFGS_action_h'] in ['LBFGS','Hessian-action-LBFGS']:
        delta_l_W = LBFGS_Hv(
                delta_l_W.t(),
                data_['Kron_LBFGS_s_y_pairs']['h'][l],
                params
            )
        delta_l_W = delta_l_W.t()
    elif params['Kron_BFGS_action_h'] == 'inv':
        Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]
        H_l_h = Kron_BFGS_matrices_l['A_inv']
        delta_l_W = torch.mm(delta_l_W, H_l_h)
    elif params['Kron_BFGS_action_h'] in ['Hessian-action-BFGS']:
        
        Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]
        H_l_h = Kron_BFGS_matrices_l['H']['h']
        delta_l_W = torch.mm(delta_l_W, H_l_h) 
            
    delta_l['W'] = delta_l_W[:, :-1]
    delta_l['b'] = delta_l_W[:, -1]
        
    return delta_l


def LBFGS_Hv(v, s_y_pairs, params):
    list_s = s_y_pairs['s']
    list_y = s_y_pairs['y']
    R_inv = s_y_pairs['R_inv']
    yTy = s_y_pairs['yTy']
    D_diag = s_y_pairs['D_diag']
    
    gamma = s_y_pairs['gamma']
    
    left_matrix = s_y_pairs['left_matrix']
    right_matrix = s_y_pairs['right_matrix']
   
    if len(list_s) == 0:
        Hv = v
    else:
        device = params['device']
        len_v_size = len(v.size())
        if len_v_size == 1:
            v = v.unsqueeze(1)
        
        if gamma == -1:
            gamma = 1 / R_inv[-1][-1].item() / yTy[-1][-1].item()
        assert gamma > 0
           
        Hv = gamma * v + torch.mm(left_matrix, torch.mm(right_matrix, v))
        if len_v_size == 1:
            Hv = Hv.squeeze(1)
    return Hv


def Kron_LBFGS_append_s_y(s, y, s_y_pairs, g_k, gamma, params):    
    s = s.unsqueeze(1)
    y = y.unsqueeze(1)
    
    device = params['device']
    
    if len(g_k) == 0:
        dot_gk_gk = 0
    else:
        dot_gk_gk = torch.mm(g_k.unsqueeze(0), g_k.unsqueeze(1)).item()

    dot_new_y_new_s = torch.mm(y.t(), s)
    dot_new_s_new_s = torch.mm(s.t(), s)
    
    if (not np.isinf(dot_new_s_new_s.item())) and\
    dot_new_y_new_s.item() > 10**(-4) * dot_new_s_new_s.item() * np.sqrt(dot_gk_gk):
        
        if len(s_y_pairs['s']) == params['Kron_BFGS_number_s_y']:
            s_y_pairs['R_inv'] = s_y_pairs['R_inv'][1:, 1:]
            
            s_y_pairs['yTy'] = s_y_pairs['yTy'][1:, 1:]
            
            s_y_pairs['s'] = s_y_pairs['s'][1:]
            s_y_pairs['y'] = s_y_pairs['y'][1:]
            
            s_y_pairs['D_diag'] = s_y_pairs['D_diag'][1:]

        if len(s_y_pairs['s']) == 0:
            s_y_pairs['s'] = s.t()
            s_y_pairs['y'] = y.t()
        else:
            
            s_y_pairs['s'] = torch.cat((s_y_pairs['s'], s.t()), dim=0)
            s_y_pairs['y'] = torch.cat((s_y_pairs['y'], y.t()), dim=0)

        if len(s_y_pairs['yTy']) == 0:
            s_y_pairs['yTy'] = torch.mm(s_y_pairs['y'], s_y_pairs['y'].t())
        else:
            yT_new_y = torch.mm(s_y_pairs['y'], y)
            
            s_y_pairs['yTy'] = torch.cat((s_y_pairs['yTy'], yT_new_y[:-1]), dim=1)
            s_y_pairs['yTy'] = torch.cat((s_y_pairs['yTy'], yT_new_y.t()), dim=0)

        if len(s_y_pairs['s']) == 1:
            s_y_pairs['D_diag'] = torch.mm(s_y_pairs['s'], s_y_pairs['y'].t())
            s_y_pairs['D_diag'] = s_y_pairs['D_diag'].squeeze(0)

            s_y_pairs['R_inv'] = 1 / s_y_pairs['D_diag'][-1]
            s_y_pairs['R_inv'] = s_y_pairs['R_inv'].unsqueeze(0).unsqueeze(1)
            
        else:
            sT_y = torch.mm(s_y_pairs['s'], y)

            s_y_pairs['D_diag'] = torch.cat(
                (s_y_pairs['D_diag'], sT_y[-1]), dim=0
            )
            
            B_22 = 1 / sT_y[-1][-1].item()
            B_22 = torch.tensor(B_22, device=device)
            
            
            B_22 = B_22.unsqueeze(0)
            B_22 = B_22.unsqueeze(1)
            
            s_y_pairs['R_inv'] = torch.cat(
                (
                    torch.cat(
                        (s_y_pairs['R_inv'], torch.zeros(1, s_y_pairs['R_inv'].size(1), device=params['device'])),
                        dim=0
                    ),
                    torch.cat((-B_22 * torch.mm(s_y_pairs['R_inv'], sT_y[:-1]), B_22), dim=0)
                ),
                dim=1
            )

        if gamma == -1:
            gamma = s_y_pairs['D_diag'][-1].item() / s_y_pairs['yTy'][-1][-1].item()
          
        
        s_y_pairs['gamma'] = gamma
        R_inv_sT = torch.mm(s_y_pairs['R_inv'], s_y_pairs['s'])
        
        if len(s_y_pairs['right_matrix']) < 2 * params['Kron_BFGS_number_s_y']:
            s_y_pairs['left_matrix'] = torch.cat(
                (R_inv_sT.t(), gamma * s_y_pairs['y'].t()), dim=1
            )
            s_y_pairs['right_matrix'] = torch.cat(
                (
                    torch.mm(torch.diag(s_y_pairs['D_diag']) + gamma * s_y_pairs['yTy'], R_inv_sT) - gamma * s_y_pairs['y'],
                    - R_inv_sT
                ), dim=0
            )
            
        else:
            m = params['Kron_BFGS_number_s_y']
            s_y_pairs['left_matrix'][:, :m] = R_inv_sT.t()
            s_y_pairs['left_matrix'][:, m:] = gamma * s_y_pairs['y'].t()
            s_y_pairs['right_matrix'][:m] = s_y_pairs['D_diag'][:, None] * R_inv_sT + gamma * (torch.mm(s_y_pairs['yTy'],R_inv_sT) - s_y_pairs['y'])
            s_y_pairs['right_matrix'][m:] = - R_inv_sT
        
    return s_y_pairs


def get_BFGS_formula(H, s, y, g_k):
    s = s.data
    y = y.data
    rho_inv = torch.dot(s, y)

    if rho_inv <= 0:
        return H, 1
    elif rho_inv <= 10**(-4) * torch.dot(s, s) * np.sqrt(torch.dot(g_k, g_k).item()):
        return H, 2
 
    rho = 1 / rho_inv

    Hy = torch.mv(H, y)
    H_new = H.data +\
    (rho**2 * torch.dot(y, torch.mv(H, y)) + rho) * torch.ger(s, s) -\
    rho * (torch.ger(s, Hy) + torch.ger(Hy, s))
    
    if torch.max(torch.isinf(H_new)):
        return H, 4
    else:
        H = H_new

    return H, 0


def train_initialization(data_, params):
    data_['Kron_LBFGS_s_y_pairs'] = {}
    if params['Kron_BFGS_action_a'] == 'LBFGS':
        L = params['numlayers']
        data_['Kron_LBFGS_s_y_pairs']['a'] = []
        for l in range(L):
            data_['Kron_LBFGS_s_y_pairs']['a'].append(
                {'s': [], 'y': [], 'R_inv': [], 'yTy': [], 'D_diag': [], 'left_matrix': [], 'right_matrix': [], 'gamma': []}
            )

    layers_params = params['layers_params']

    device = params['device']
    numlayers = params['numlayers']
    model = data_['model']
        
    data_['Kron_BFGS_momentum_s_y'] = []
    for l in range(numlayers):
        Kron_BFGS_momentum_s_y_l = {}
        Kron_BFGS_momentum_s_y_l['s'] = torch.zeros(layers_params[l]['output_size'], device=device)
        Kron_BFGS_momentum_s_y_l['y'] = torch.zeros(layers_params[l]['output_size'], device=device)
        data_['Kron_BFGS_momentum_s_y'].append(Kron_BFGS_momentum_s_y_l)

    data_['Kron_BFGS_matrices'] = []
    for l in range(numlayers):
        Kron_BFGS_matrices_l = {}
        size_A = layers_params[l]['input_size'] + 1 
        Kron_BFGS_matrices_l['A'] = torch.zeros(size_A, size_A, device=device, requires_grad=False)
        data_['Kron_BFGS_matrices'].append(Kron_BFGS_matrices_l)
    # exact estimation of post-activation matrix A
    model.eval()
    if params['N1'] < params['num_train_data']*params['N1']:        
        for batch, (X, y) in enumerate(data_['dataset']): 
            torch.cuda.empty_cache()
            X.to(device)
            z = model.forward(X)
            a = model.a
            h = model.h

            a, h = reshape_a_h(a, h, layers_params)

            j = batch + 1

            for l in range(numlayers):
                h_l = h[l]
                if layers_params[l]['name'] == 'conv':
                    ones = torch.ones(h_l.size()[0], 1, device=device)
                    h_l = torch.cat([*[h_l for _ in range(1)], ones], dim=1)
                    A_j = torch.matmul(torch.t(h_l), h_l) / h_l.size(0)
                elif layers_params[l]['name'] == 'fc':
                    ones = torch.ones(h_l.size()[0], 1, device=device)
                    h_l = torch.cat([h_l.data, ones], dim=1)
                    A_j = torch.matmul(torch.t(h_l), h_l) / h_l.size(0)
                data_['Kron_BFGS_matrices'][l]['A'] *= (j-1)/j
                data_['Kron_BFGS_matrices'][l]['A'] += 1/j * A_j

            if params['debug']:
                break    

    return data_, params


def reshape_a_h(a, h, layers_params, kernel_size=3, stride=1):
    a_new = []
    h_new = []
    for l in range(len(a)):
        if layers_params[l]['name'] == 'conv':
            h_l = h[l].unfold(dimension=3, size=kernel_size, step=stride)
            h_l = h_l.unfold(2, kernel_size, stride)
            h_l = h_l.mean(dim=[2,3])
            h_l = h_l.reshape(-1, layers_params[l]['input_size'])
            a_l = a[l]
            a_l = a_l.mean(dim=[2,3])
        elif layers_params[l]['name'] == 'fc':
            h_l = h[l]
            a_l = a[l]
        a_new.append(a_l)
        h_new.append(h_l)
    return a_new, h_new


def get_second_order_caches(z, a, h, data_, params):
    N1 = params['N1']
    N2 = params['N2']

    N2_index = np.random.permutation(N1)[:N2]
    params['N2_index'] = N2_index

    X_mb = data_['X_mb']

    data_['X_mb_N1'] = X_mb
    X_mb_N2 = X_mb[N2_index]
    data_['X_mb_N2'] = X_mb_N2

    # Empirical Fisher
    t_mb = data_['t_mb']
    data_['t_mb_pred_N2'] = t_mb[N2_index]
    # collect pre-activations gradients
    data_['a_grad_N2'] = [N2 * (a_l.grad)[N2_index] for a_l in a]
    # reshape a and h
    a, h, = reshape_a_h(a, h, params['layers_params'])
    data_['h_N2'] = [h_l[N2_index].data for h_l in h]
    data_['a_N2'] = [a_l[N2_index].data for a_l in a]
    # reshape gradients
    for l in range(params['numlayers']):
        if params['layers_params'][l]['name'] == 'conv':
            data_['a_grad_N2'][l] = data_['a_grad_N2'][l].mean(dim=[2,3])

    return data_


def update_parameter(p_torch, model, params):
    numlayers = params['numlayers']
    alpha = params['alpha']
    device = params['device']

    for l in range(numlayers):
        if params['layers_params'][l]['name'] == 'conv':
            model.layers_weights[l]['W'].data += alpha * p_torch[l]['W'].data
            model.layers_weights[l]['b'].data += alpha * p_torch[l]['b'].data
        elif params['layers_params'][l]['name'] == 'fc':
            model.layers_weights[l]['W'].data += alpha * p_torch[l]['W'].data
            model.layers_weights[l]['b'].data += alpha * p_torch[l]['b'].data
        
    return model

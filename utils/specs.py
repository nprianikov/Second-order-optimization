import os
import sys
sys.path.append("..")

import copy
import json
import numpy as np
import torch 
from torchvision import datasets, transforms
from utils.pyhessian import hessian # Hessian computation
from utils.density_plot import get_esd_plot # ESD computation
import src.experiments_maker as experiments_maker
import src.engine as engine
import utils.config_manager as cm
from utils.pyhessian.utils import normalization

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


class CheckpointSpecs:
    def __init__(self, f, full_dataset=False, pref=None, **kwargs):
        self._f = f
        self._file_name = self._f.split('/')[-1].split('.')[0]
        if pref is not None:
            self._file_name = pref + '_' + '_'.join(self._file_name.split('_')[1:])
        self._full_dataset = full_dataset
        # read the dict from txt file
        # config_path = os.path.join(f, '..', 'config.txt')
        config_path = '/'.join(f.split('/')[:-1]) + '/config.txt'
        with open(config_path, 'r') as file:
            data = file.read()
        self._config = json.loads(data)
        #self._config['batch_size'] = 32
        torch.cuda.empty_cache()
        self._model, self._train_data_loader, self._test_data_loader, self._optimizer, self._accuracy_fn = experiments_maker.make(self._config, cm.device, **kwargs)
        self._model.load_state_dict(torch.load(f))#, map_location=cm.device))
        self._model.eval()

        self._cuda = True if cm.device == torch.device('cuda') else False
        self._cuda = True
        # for batch, (X, y) in enumerate(self._train_data_loader):
        #    # Send data to GPU
        #    X, y = X.to(cm.device), y.to(cm.device)
        #
        #for batch, (X, y) in enumerate(self._train_data_loader):
        #    # Send data to GPU
        #    X, y = X.to(cm.device), y.to(cm.device)
        
        self._inputs, self._targets = next(iter(self._train_data_loader))
        self._inputs = self._inputs.to(cm.device)
        self._targets = self._targets.to(cm.device)
        if full_dataset:
            self.hessian_comp = hessian(self._model, cm.loss_fn, dataloader=self._train_data_loader, cuda=self._cuda)
        else:
            self.hessian_comp = hessian(self._model, cm.loss_fn, data=(self._inputs, self._targets), cuda=self._cuda)
        self.top_eigenvalues, self.top_eigenvector = self.hessian_comp.eigenvalues(maxIter=100, tol=1e-2, top_n=2)
        self.trace = self.hessian_comp.trace()


    def copy_model(self):
        # create a copy of the model
        model_perb = copy.deepcopy(self._model)
        model_perb.load_state_dict(torch.load(self._f))
        model_perb.eval()
        return model_perb


    def continue_training(self):
        engine.train(self._model, self._train_data_loader, self._test_data_loader, cm.loss_fn, self._optimizer, self._accuracy_fn, cm.device, self._config)


    def perb_params_1d(self, model_orig,  model_perb, direction, alpha):
        for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
            m_perb.data = m_orig.data + alpha * d
        return model_perb


    def perb_params_2d(self, model_orig, model_perb, directions, alphas):
        for m_orig, m_perb, d1, d2 in zip(model_orig.parameters(), model_perb.parameters(), directions[0], directions[1]):
            m_perb.data = m_orig.data + alphas[0] * d1 + alphas[1] * d2
        return model_perb


    def empirical_test_loss(self, model, slice_size=0.25):
        test_loss = 0
        test_acc = 0
        slice_len = int(len(self._test_data_loader) * slice_size)
        for count, (inputs, targets) in enumerate(self._test_data_loader):
            if count >= slice_len:
                break
            outputs = model(inputs)
            loss = cm.loss_fn(outputs, targets)
            test_loss += loss
            test_acc += self._accuracy_fn(outputs.argmax(dim=1), targets).item()
        test_loss = test_loss / len(slice_len)
        test_acc = test_acc / len(slice_len)
        return test_loss, test_acc


    def empirical_train_loss(self, model, slice_size=0.25):
        train_loss = 0
        train_acc = 0
        slice_len = int(len(self._train_data_loader) * slice_size)
        for count, (inputs, targets) in enumerate(self._train_data_loader):
            if count >= slice_len:
                break
            inputs = inputs.to(cm.device)
            targets = targets.to(cm.device)
            outputs = model(inputs)
            loss = cm.loss_fn(outputs, targets)
            train_loss += loss
            train_acc += self._accuracy_fn(outputs.argmax(dim=1), targets).item()
            #print(f'count: {count}, loss: {loss}')
        train_loss = train_loss / len(slice_len)
        train_acc = train_acc / len(slice_len)
        return train_loss, train_acc

    def perb_loss_1d(self, direction, radius=5e-1, n=21, **kwargs):
        lams = np.linspace(-radius, radius, n).astype(np.float32)
        loss_list = []
        model_perb = self.copy_model()
        for lam in lams:
            model_perb = self.perb_params_1d(self._model, model_perb, direction, lam)
            if self._full_dataset:
                loss, _ = self.empirical_train_loss(model_perb)
            else:
                loss = cm.loss_fn(model_perb(self._inputs), self._targets).item() 
            loss_list.append(loss)
        # plot y horizontal line for point with min loss
        plt.clf()
        plt.axhline(y=min(loss_list), color='r', linestyle='--')
        plt.plot(lams, loss_list)
        # display min loss on y axis
        ytickValues = plt.yticks()[0]
        new_ytick_value = min(loss_list)
        ytick_values = np.append(ytickValues, new_ytick_value)
        ytick_labels = [str(round(val, 2)) for val in ytick_values]
        # plot
        plt.yticks(ytick_values, ytick_labels)
        plt.ylabel('Loss')
        plt.xlabel('Perturbation')
        # plt.title('Loss landscape perturbed based on top Hessian eigenvector')


    def perb_loss_2d(self, directions, radius=5e-1, n=21, **kwargs):
        alphas = np.meshgrid(np.linspace(-radius, radius, n).astype(np.float32),
                            np.linspace(-radius, radius, n).astype(np.float32))
        loss_list = []
        model_perb = self.copy_model()
        for i in range(len(alphas[0])):
            for j in range(len(alphas[1])):
                model_perb = self.perb_params_2d(self._model, model_perb, directions=directions, alphas=[alphas[0][i][j], alphas[1][i][j]])
                if self._full_dataset:
                    loss, _ = self.empirical_train_loss(model_perb)
                else:
                    loss = cm.loss_fn(model_perb(self._inputs), self._targets).item() 
                loss_list.append(loss)
        # Create a 3D plot
        plt.clf()
        fig = plt.figure()
        plt.rcParams.update({'font.size': 8})
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(alphas[1], alphas[0], np.array(loss_list).reshape(alphas[0].shape), cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('Alpha 2')
        ax.set_ylabel('Alpha 1')
        ax.set_zlabel('Loss')
        ax.view_init(elev=55, azim=45)
        # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.6, 1]))


    def get_flat_params(self):
        model = self.copy_model()
        params = [p.data.cpu().numpy().flatten() for _, p in model.named_parameters()]
        names = [n for n, _ in model.named_parameters()]
        result = zip(names, params)
        return result
    

    def compute_grads(self, full_dataset=False):
        if full_dataset:
            loss, _ = self.empirical_train_loss(self._model)
        else:
            loss = cm.loss_fn(self._model(self._inputs), self._targets)
        loss.backward()


    def get_flat_grad(self):
        model = self.copy_model()
        grads = [p.grad.data.cpu().numpy().flatten() for _, p in model.named_parameters()]
        names = [n for n, _ in model.named_parameters()]
        result = zip(names, grads)
        return result


    def combined_hists(self, data, title, xlabel, ylabel='Density', bins=100, **kwargs):
        '''
        Create a combined plot of histograms for provided data per layer e.g. weights, gradients, etc.
        :data - list of tuples (name, param) where param is a tensor
        '''
        plt.clf()
        param_list = list(data)
        fig, axs = plt.subplots(nrows=len(param_list)//2, ncols=2)
        axs = axs.flatten()
        for count, (name, param) in enumerate(param_list):
            ax = axs[count]
            ax.hist(param, bins=bins)
            ax.set_title(name, fontsize=8)

            for label in ax.get_xticklabels():
                label.set_fontsize(8)
            for label in ax.get_yticklabels():
                label.set_fontsize(8) 

        fig.suptitle(title)
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f"combined_hists_{self._file_name}_{xlabel}.png")
        plt.show()


    def combined_log_density(self, data, title, xlabel, ylabel='Density (Log Scale)', bins=1000, **kwargs):
        '''
        Create a combined plot of log density for provided data per layer e.g. weights, gradients, etc.
        :data - list of tuples (name, param) where param is a tensor
        '''
        plt.clf()
        param_list = list(data)
        fig, axs = plt.subplots(nrows=len(param_list)//2, ncols=2)
        axs = axs.flatten()
        for count, (name, param) in enumerate(param_list):
            ax = axs[count]
            
            log_density, bin_edges = np.histogram(param, bins=bins, density=False)
            log_density = np.log(log_density)
            ax.plot(bin_edges[1:], log_density)

            ax.set_title(name, fontsize=8)
            for label in ax.get_xticklabels():
                label.set_fontsize(8)
            for label in ax.get_yticklabels():
                label.set_fontsize(8)
        fig.suptitle(title)
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f"combined_log_density_{self._file_name}_{xlabel}.png")
        plt.show()


    def full_log_density(self, data, title, xlabel, ylabel='Density (Log Scale)', bins=1000, **kwargs):
        '''
        Create a single plot of log density for provided data e.g. weights, gradients, etc.
        :data - list of tuples (name, param) where param is a tensor
        '''
        plt.clf()
        param_list = list(data)
        data_points = np.array([])
        for _, (_, param) in enumerate(param_list):
            data_points = np.append(data_points, param)
        log_density, bin_edges = np.histogram(data_points, bins=bins, density=True)
        log_density = np.log(log_density)
        plt.plot(bin_edges[1:], log_density)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"full_log_density{self._file_name}_{xlabel}.png")
        plt.show()


    def full_hist(self, data, title, xlabel, ylabel='Density', bins=1000, **kwargs):
        '''
        Create a single histogram plot for provided data e.g. weights, gradients, etc.
        :data - list of tuples (name, param) where param is a tensor
        '''
        plt.clf()
        param_list = list(data)
        data_points = np.array([])
        for _, (_, param) in enumerate(param_list):
            data_points = np.append(data_points, param)
        plt.hist(data_points, bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"full_hists_{self._file_name}_{xlabel}.png")
        plt.show()


    def params_hist(self, bins=100, **kwargs):
        self.combined_hists(self.get_flat_params(), title=self._file_name, xlabel='Parameters', bins=bins, **kwargs)


    def params_log_density(self, bins=1000, **kwargs):
        self.combined_log_density(self.get_flat_params(), title=self._file_name, xlabel='Parameters', bins=bins, **kwargs)


    def full_params_hist(self, bins=100, **kwargs):
        self.full_hist(self.get_flat_params(), title=self._file_name, xlabel='Parameters', bins=bins, **kwargs)

    
    def full_params_log_density(self, bins=1000, **kwargs):
        self.full_log_density(self.get_flat_params(), title=self._file_name, xlabel='Parameters', bins=bins, **kwargs)


    def grads_hist(self, bins=100, **kwargs):
        '''
        Requires a model with gradients computed
        '''
        self.combined_hists(self.get_flat_grad(), title=self._file_name, xlabel='Gradients', bins=bins, **kwargs)


    def grads_log_density(self, bins=1000, **kwargs):
        '''
        Requires a model with gradients computed
        '''
        self.combined_log_density(self.get_flat_grad(), title=self._file_name, xlabel='Gradients', bins=bins, **kwargs)


    def full_grads_hist(self, bins=100, **kwargs):
        '''
        Requires a model with gradients computed
        '''
        self.full_hist(self.get_flat_grad(), title=self._file_name, xlabel='Gradients', bins=bins, **kwargs)


    def full_grads_log_density(self, bins=1000, **kwargs):
        '''
        Requires a model with gradients computed
        '''
        self.full_log_density(self.get_flat_grad(), title=self._file_name, xlabel='Gradients', bins=bins, **kwargs)

 
    def top_eig_perb_plot(self, radius=5e-1, n=21, **kwargs):
        self.perb_loss_1d(self.top_eigenvector[0], radius=radius, n=n, **kwargs)
        plt.savefig(f"top_eig_perb_plot_{self._file_name}_{radius}.png")
        plt.show()


    def top_2_eig_perb_plot(self, radius=5e-1, n=21, **kwargs):
        self.perb_loss_2d(self.top_eigenvector[:2], radius=radius, n=n, **kwargs)
        plt.savefig(f"top_2_eig_perb_plot_{self._file_name}_{radius}.png")
        plt.show()


    def rand_perb_plot(self, radius=5e-1, n=21, **kwargs):
        v = normalization([torch.randn_like(p) for p in self._model.parameters()])
        self.perb_loss_1d(v, radius=radius, n=n, **kwargs)
        plt.savefig(f"rand_perb_plot_{self._file_name}_{radius}.png")
        plt.show()


    def rand_2d_perb_plot(self, radius=5e-1, n=21, **kwargs):
        v1 = normalization([torch.randn_like(p) for p in self._model.parameters()])
        v2 = normalization([torch.randn_like(p) for p in self._model.parameters()])
        self.perb_loss_2d([v1, v2], radius=radius, n=n, **kwargs)
        plt.savefig(f"rand_2d_perb_plot_{self._file_name}_{radius}.png")
        plt.show()


    def grad_perb_plot(self, radius=5e-1, n=21, **kwargs):
        model_perb = self.copy_model()
        if self._full_dataset:
            loss, _ = self.empirical_train_loss(model_perb)
        else:
            loss = cm.loss_fn(model_perb(self._inputs), self._targets)
        loss.backward()

        v = [p.grad.data for p in model_perb.parameters()]
        v = normalization(v)
        model_perb.zero_grad()

        self.perb_loss_1d(v, radius=radius, n=n, **kwargs)
        plt.savefig(f"grad_perb_plot_{self._file_name}_{radius}.png")
        plt.show()


    def hessian_esd_plot(self, save=False, **kwargs):
        plt.clf()
        trace = self.hessian_comp.trace()
        print("The trace of this model is: %.4f"%(np.mean(trace)))
        density_eigen, density_weight = self.hessian_comp.density() 
        save_path = f"hessian_esd_plot_{self._file_name}.png" if save else None
        get_esd_plot(density_eigen, density_weight, save_path=save_path, **kwargs)


    def layer_params_plot(self, **kwargs):
        '''
        Requires precomputed gradients
        '''
        model = self._model
        #
        def add_id_to_lists(A, B, x, s):
            C = []
            for i in range(len(A)):
                C.append([A[i], B[i], x, s])
            return C
        #
        ns = []
        vars = []
        parameter_position_pairs = []
        for param_idx, (name, param) in enumerate(model.named_parameters()):
            params = list(param.data.cpu().numpy().flatten())
            gradients = list(param.grad.data.cpu().numpy().flatten())
            scaler = np.var(params)
            ns.append(name)
            vars.append(scaler)
            id_patams = add_id_to_lists(params, gradients, param_idx, scaler)
            parameter_position_pairs.extend(id_patams)
        #
        min_marker_size = 5
        max_marker_size = 30
        #
        xs = [x[2] for x in parameter_position_pairs]
        ys = [x[0] for x in parameter_position_pairs]
        zs = [x[1] for x in parameter_position_pairs]
        vs = [x[3] for x in parameter_position_pairs]
        marker_sizes = min_marker_size + (max_marker_size - min_marker_size) * (vs - np.min(vars)) / (np.max(vars) - np.min(vars))
        #
        jitter_amount = 0.3
        x_jitter = np.random.uniform(-jitter_amount, jitter_amount, len(xs))
        # 
        plt.scatter(x=xs+x_jitter, y=ys, s=marker_sizes, vmin=np.min(zs), vmax=np.max(zs), alpha=0.5, c=zs, cmap='seismic')
        plt.ylabel('Parameter values')
        plt.xlabel('Layer')
        plt.title(f'Parameters per layer: stratified by gradient values')
        plt.xticks(list(range(0, len(ns))), ns, rotation=60)
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"layer_params_{self._file_name}.png")
        plt.show()

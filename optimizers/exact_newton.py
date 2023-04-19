import torch
from torch.optim import Optimizer
from functorch import hessian
import torch.nn.functional as F
from functorch import jacrev
from functorch import jacrev, jacfwd


# https://github.com/pytorch/pytorch/blob/v2.0.0-rc6/torch/optim/optimizer.py
# https://www.youtube.com/watch?v=zvp8K4iX2Cs


class Newton(Optimizer):
    def __init__(self, params, lr=1.0, eps=1e-6, max_iter=100):
        defaults = dict(lr=lr, eps=eps, max_iter=max_iter)
        super(Newton, self).__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise ValueError('closure should be provided')

        # Compute gradients and Hessian matrix
        loss = closure()
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        flat_grads = torch.cat([grad.view(-1) for grad in grads])
        hessian = torch.zeros(flat_grads.size()[0], flat_grads.size()[0])
        for i in range(flat_grads.size()[0]):
            grad2 = torch.autograd.grad(grads[0][i], self.params, retain_graph=True)[0]
            hessian[i] = torch.cat([g.view(-1) for g in grad2])

        # Solve the Newton system to get the search direction
        search_dir = torch.matmul(torch.inverse(hessian), flat_grads)
        search_dir = search_dir.view_as(grads[0])

        # Update the parameters
        for i, param in enumerate(self.params):
            param.data.add_(-self.defaults['lr'], search_dir[i])

# -*- coding: utf-8 -*-
"""
SGLD
"""
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

class SGLD(Optimizer):
    """
    Implements stochastic gradient Langevin dynamics
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups/
        lr (float): learning rate.
        momentum (float, optional): momentum factor (default: 0).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        dampening (float, optional): dampening for momentum (default: 0).
        nesterov (bool, optional): enables Nesterov momentum (default: False).
    """
    def __init__(self, params, lr=required, momentum=0., dampening=0., weight_decay=0., nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGLD, self).__init__(params, defaults)
        
        sizes = []
        for param in self.param_groups[0]['params']:
            sizes.append(torch.numel(param))
            
        self.lr = lr
        self.sizes = sizes
    
    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            # Update the parameters
            for param in group['params']:
              #if idx < 1:
                if param.grad is None:
                    continue
                
                langevin_noise = param.data.new(param.data.size()).normal_(mean=0, std=1)*np.sqrt(2*group['lr'])
                
                d_p = param.grad.data
                
                if weight_decay != 0:
                    d_p.add_(weight_decay, param.data)
                
                if momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(param.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                param.data.add_(-group['lr'], d_p + langevin_noise)
        
        return loss
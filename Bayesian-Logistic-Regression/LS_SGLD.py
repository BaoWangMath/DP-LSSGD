# -*- coding: utf-8 -*-
"""
LS-SGLD
"""
import numpy as np
import torch
import random
from Grad_optimizer import Optimizer, required
from torch.autograd import Variable
import pytorch_fft.fft as fft
import time
import math

class LS_SGLD(Optimizer):
    """
    Implements stochastic gradient Langevin dynamics with Laplacian smoothing
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups/
        lr (float): learning rate.
        sigma (float, optional): Laplacian smoothing parameter.
        momentum (float, optional): momentum factor (default: 0).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        dampening (float, optional): dampening for momentum (default: 0).
        nesterov (bool, optional): enables Nesterov momentum (default: False).
    """
    def __init__(self, params, lr=required, sigma=1., momentum=0., dampening=0., weight_decay=0., nesterov=False):
        defaults = dict(lr=lr, sigma=sigma, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LS_SGLD, self).__init__(params, defaults)
        
        sizes = []
        for param in self.param_groups[0]['params']:
            sizes.append(torch.numel(param))
        
        coeffs = []
        zero_Ns = []
        for size in sizes:
            '''
            c = np.zeros(shape=(1, size))
            c[0, 0] = -2.; c[0, 1] = 1.; c[0, -1] = 1.
            c = torch.Tensor(c).cuda()
            zero_N = torch.zeros(1, size).cuda()
            c_fft = torch.rfft(c, 1, onesided=False) # This is wrong
            coeff = 1./(1.-sigma*c_fft)
            coeffs.append(coeff)
            zero_Ns.append(zero_N)
            '''
            c = np.zeros(shape=(1, size))
            c[0, 0] = -2.
            c[0, 1] = 1.
            c[0, -1] = 1.
            c = torch.Tensor(c).cuda()
            zero_N = torch.zeros(1, size).cuda()
            c_fft, _ = fft.fft(c, zero_N)
            coeff = 1. / (1.-sigma*c_fft)
            coeffs.append(coeff)
            zero_Ns.append(zero_N)
            
        self.lr = lr
        self.sigma = sigma
        self.sizes = sizes
        self.coeffs = coeffs
        self.zero_Ns = zero_Ns
    
    def __setstate__(self, state):
        super(LS_SGLD, self).__setstate__(state)
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
            idx = 0
            for param in group['params']:
              #if idx < 1:
                if param.grad is None:
                    continue
                tmp = param.grad.view(-1, self.sizes[idx])
                
                # The standard deviation of the injected noise
                eps = math.sqrt(2.*self.lr)
                
                
                tmp = tmp.data + eps*torch.randn(tmp.shape).cuda()
                re, im = fft.fft(tmp, self.zero_Ns[idx])
                re = re*self.coeffs[idx]
                im = im*self.coeffs[idx]
                tmp = fft.ifft(re, im)[0]
                # tmp = Variable(tmp)  # Xiyang : I think this slows it down a little.
                tmp = tmp.view(param.grad.size())
                param.grad.data = tmp
                #print(p.grad)
                idx += 1
                
                
                '''
                tmp1 = tmp.data
                tmp2 = eps*torch.randn(tmp.shape).cuda()
                ft_tmp = torch.rfft(tmp2, 1, onesided = False)
                tmp2 = torch.mul(ft_tmp,self.coeffs[idx])
                tmp2 = torch.irfft(tmp2,1,onesided = False)
                tmp2 = tmp2.view(param.grad.size())
                tmp1 = tmp1.view(param.grad.size())
                param.grad.data = tmp1 + tmp2
                idx += 1
                '''
                
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
                
                param.data.add_(-group['lr'], d_p)
        
        return loss
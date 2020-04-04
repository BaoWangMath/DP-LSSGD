# -*- coding: utf-8 -*-
"""
LS SGLD 2
Noise term A^{-1/2}
"""
import numpy as np
import torch
import random
from Grad_optimizer import Optimizer, required
from torch.autograd import Variable
import pytorch_fft.fft as fft
import time
import math

class LSSGLD2(Optimizer):
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
        vecs: vectors that used to represent 
    """
    def __init__(self, params, lr=required, sigma=1., momentum=0., dampening=0., weight_decay=0., nesterov=False, vecs=None):
        defaults = dict(lr=lr, sigma=sigma, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, vecs=vecs)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LSSGLD2, self).__init__(params, defaults)
        
        coeffs_noise = []
        
        for vec in vecs:
            c = torch.Tensor(vec).cuda()
            zero_N = torch.zeros(1, vec.shape[1]).cuda()
            c_fft, _ = fft.fft(c, zero_N)
            coeff = 1./c_fft
            coeffs_noise.append(coeff)
        
        sizes = []
        for param in self.param_groups[0]['params']:
            sizes.append(torch.numel(param))
        
        coeffs = []
        zero_Ns = []
        for size in sizes:
            if size > 2:
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
        self.coeffs_noise = coeffs_noise
        
    
    def __setstate__(self, state):
        super(LSSGLD2, self).__setstate__(state)
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
              if self.sizes[idx] > 2:
              #if idx < 1:
                if param.grad is None:
                    continue
                tmp = param.grad.view(-1, self.sizes[idx])
                
                # The standard deviation of the injected noise
                eps = math.sqrt(2.*self.lr)
                
                tmp1 = tmp.data
                tmp2 = eps*torch.randn(tmp.shape).cuda()
                
                re, im = fft.fft(tmp1, self.zero_Ns[idx])
                re, im = re*self.coeffs[idx], im*self.coeffs[idx]
                tmp1 = fft.ifft(re, im)[0]
                
                re, im = fft.fft(tmp2, self.zero_Ns[idx])
                re, im = re*self.coeffs_noise[idx], im*self.coeffs_noise[idx]
                tmp2 = fft.ifft(re, im)[0]
                
                tmp = tmp1 + tmp2
                tmp = tmp.view(param.grad.size())
                
                param.grad.data = tmp
                #print(p.grad)
                idx += 1
                
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
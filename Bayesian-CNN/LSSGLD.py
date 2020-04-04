# -*- coding: utf-8 -*-
"""
LS SGLD 2
Noise term A^{-1/2}
"""
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
#import pytorch_fft.fft as fft
import math

class LSSGLD(Optimizer):
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
        vecs: vectors that used to represent the sqrt of the Laplacian smoothing matrix.
    """
    def __init__(self, params, lr=required, sigma=1., momentum=0., dampening=0., weight_decay=0., nesterov=False, vecs=None):
        defaults = dict(lr=lr, sigma=sigma, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, vecs=vecs)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LSSGLD, self).__init__(params, defaults)
        
        fft_vec_noise = []
        
        for vec in vecs:
            c = torch.Tensor(vec).cuda()
            zero_N = torch.zeros(1, vec.shape[1]).cuda()
            #c_fft, _ = fft.fft(c, zero_N)
            c_fft = torch.rfft(c, 1, onesided=False)#.to(device)
            #coeff = 1./c_fft
            coeff = 1./c_fft[...,0]
            fft_vec_noise.append(coeff)
        
        sizes = []
        for param in self.param_groups[0]['params']:
            sizes.append(torch.numel(param))
        
        fft_vec_grad = []
        zero_Ns = []
        for size in sizes:
            if size > 2:
               c = np.zeros(shape=(1, size))
               c[0, 0] = -2.
               c[0, 1] = 1.
               c[0, -1] = 1.
               c = torch.Tensor(c).cuda()
               zero_N = torch.zeros(1, size).cuda()
               #c_fft, _ = fft.fft(c, zero_N)
               c_fft = torch.rfft(c, 1, onesided=False)
               #coeff = 1. / (1.-sigma*c_fft)
               coeff = 1./(1.-sigma*c_fft[...,0])
               fft_vec_grad.append(coeff)
               zero_Ns.append(zero_N) 
        
        self.lr = lr
        self.sigma = sigma
        self.sizes = sizes
        self.fft_vec_grad = fft_vec_grad
        self.zero_Ns = zero_Ns
        self.fft_vec_noise = fft_vec_noise
        
    
    def __setstate__(self, state):
        super(LSSGLD, self).__setstate__(state)
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
                
                '''
                re, im = fft.fft(tmp1, self.zero_Ns[idx])
                re, im = re*self.fft_vec_grad[idx], im*self.fft_vec_grad[idx]
                tmp1 = fft.ifft(re, im)[0]
                '''
                ft_tmp1 = torch.rfft(tmp1, 1, onesided=False)
                tmp1 = torch.zeros_like(ft_tmp1)
                tmp1[...,0] = ft_tmp1[...,0]*self.fft_vec_grad[idx]
                tmp1[...,1] = ft_tmp1[...,1]*self.fft_vec_grad[idx]
                tmp1 = torch.irfft(tmp1, 1, onesided = False)
                
                '''
                re, im = fft.fft(tmp2, self.zero_Ns[idx])
                re, im = re*self.fft_vec_noise[idx], im*self.fft_vec_noise[idx]
                tmp2 = fft.ifft(re, im)[0]
                '''
                ft_tmp2 = torch.rfft(tmp2, 1, onesided=False)
                tmp2 = torch.zeros_like(ft_tmp2)
                tmp2[...,0] = ft_tmp2[...,0]*self.fft_vec_noise[idx]
                tmp2[...,1] = ft_tmp2[...,1]*self.fft_vec_noise[idx]
                tmp2 = torch.irfft(tmp2, 1, onesided = False)
                
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
              else:
                if param.grad is None:
                    continue
                tmp = param.grad.view(-1, self.sizes[idx])
                
                # The standard deviation of the injected noise
                eps = math.sqrt(2.*self.lr)
                
                tmp1 = tmp.data
                tmp2 = eps*torch.randn(tmp.shape).cuda()
                
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


'''
# -*- coding: utf-8 -*-
"""
LS SGLD
"""
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
import pytorch_fft.fft as fft


class LSSGLD(Optimizer):
    """
    Implements Laplacian smoothing stochastic gradient Langevin dynamics
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups/
        lr (float): learning rate.
        momentum (float, optional): momentum factor (default: 0).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        dampening (float, optional): dampening for momentum (default: 0).
        nesterov (bool, optional): enables Nesterov momentum (default: False).
        sigma: Laplacian smoothing constant
        vecs: the vectors represents sqrt of the Laplacian smoothing matrix
    """
    def __init__(self, params, lr=required, momentum=0., dampening=0., weight_decay=0., nesterov=False, sigma=1., vecs=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, sigma=sigma, vecs=vecs)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LSSGLD, self).__init__(params, defaults)
        
        fft_vec_noise = []
        for vec in vecs:
            c = torch.Tensor(vec).cuda()
            zero_N = torch.zeros(1, vec.shape[1]).cuda()
            c_fft, _ = fft.fft(c, zero_N)
            coeff = 1./c_fft
            fft_vec_noise.append(coeff)
        
        sizes = []
        for param in self.param_groups[0]['params']:
            sizes.append(torch.numel(param))
        
        fft_vec_grad = []
        zero_Ns = []
        for size in sizes:
            if size > 2:
                c = np.zeros(shape=(1, size))
                c[0, 0] = -2.; c[0, 1] = 1.; c[0, -1] = 1.
                c = torch.Tensor(c).cuda()
                zero_N = torch.zeros(1, size).cuda()
                c_fft, _ = fft.fft(c, zero_N)
                coeff = 1. / (1.-sigma*c_fft)
                fft_vec_grad.append(coeff)
                zero_Ns.append(zero_N) 
        
        self.lr = lr
        self.sigma = sigma
        self.sizes = sizes
        self.fft_vec_noise = fft_vec_noise
        self.fft_vec_grad = fft_vec_grad
        self.zero_Ns = zero_Ns
        
    
    def __setstate__(self, state):
        super(LSSGLD, self).__setstate__(state)
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
              if self.sizes[idx] > 2: # Need Laplacian smoothing
                if param.grad is None:
                    continue
                
                # Laplacian smoothing the gradient
                tmp = param.grad.view(-1, self.sizes[idx])
                tmp1 = tmp.data
                re, im = fft.fft(tmp1, self.zero_Ns[idx])
                re, im = re*self.fft_vec_grad[idx], im*self.fft_vec_grad[idx]
                tmp1 = fft.ifft(re, im)[0]
                
                # Laplacian smoothing the noise
                tmp2 = np.sqrt(2*group['lr'])*torch.randn(tmp.shape).cuda()
                re, im = fft.fft(tmp2, self.zero_Ns[idx])
                re, im = re*self.fft_vec_noise[idx], im*self.fft_vec__noise[idx]
                tmp2 = fft.ifft(re, im)[0]
                
                tmp = tmp1 + tmp2
                tmp = tmp.view(param.grad.size())
                
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
'''
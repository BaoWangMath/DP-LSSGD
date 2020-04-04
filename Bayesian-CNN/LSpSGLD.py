# -*- coding: utf-8 -*-
"""
Laplacian smoothing rmsprop_SGLD, i.e., LSpSGLD
"""
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
#import pytorch_fft.fft as fft
import math


class LSpSGLD(Optimizer):
    """Implements LSpSGLD algorithm.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered pSGLD,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, sigma=1., vecs=None):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, sigma=sigma, vecs=vecs)
        super(LSpSGLD, self).__init__(params, defaults)
        
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
        
        self.sigma = sigma
        self.sizes = sizes
        self.fft_vec_grad = fft_vec_grad
        self.zero_Ns = zero_Ns
        self.fft_vec_noise = fft_vec_noise
        
    
    def __setstate__(self, state):
        super(LSpSGLD, self).__setstate__(state)
    
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            idx = 0
            for p in group['params']:
              if self.sizes[idx] > 2:
                if p.grad is None:
                    continue
                
                # Perform LS on the gradient
                tmp = p.grad.view(-1, self.sizes[idx])
                tmp1 = tmp.data
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
                
                p.grad.data = tmp1.view(p.grad.size())
                # Compute the preconditioning matrix
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                
                state['step'] += 1
                
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                # Laplacian smoothing for Langevin noise
                langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) * np.sqrt(2*group['lr'])
                tmp2 = langevin_noise.view(-1, self.sizes[idx])
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
                langevin_noise = tmp2.view(p.grad.size())
                
                idx += 1
                # Langevin noise
                
                #print 'T1: ', langevin_noise.type
                #print 'T2: ', p.grad.data.type
                
                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                avg = square_avg.sqrt().add_(group['eps'])
                #p.data.addcdiv_(-group['lr'], grad, avg)
                p.data.add_(-group['lr'], grad.div_(avg) + langevin_noise/torch.sqrt(avg) )
              else:
                if p.grad is None:
                    continue
                
                # Perform LS on the gradient
                tmp = p.grad.view(-1, self.sizes[idx])
                tmp1 = tmp.data
                p.grad.data = tmp1.view(p.grad.size())
                # Compute the preconditioning matrix
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                
                state['step'] += 1
                
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                # Laplacian smoothing for Langevin noise
                langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) * np.sqrt(2*group['lr'])
                tmp2 = langevin_noise.view(-1, self.sizes[idx])
                langevin_noise = tmp2.view(p.grad.size())
                
                idx += 1
                # Langevin noise
                
                #print 'T1: ', langevin_noise.type
                #print 'T2: ', p.grad.data.type
                
                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                avg = square_avg.sqrt().add_(group['eps'])
                #p.data.addcdiv_(-group['lr'], grad, avg)
                p.data.add_(-group['lr'], grad.div_(avg) + langevin_noise/torch.sqrt(avg) )

        return loss
"""
LSSGD
"""
import numpy as np
import torch
import random
#import pytorch_fft.fft as fft
from Grad_optimizer import Optimizer, required
from torch.autograd import Variable
import time

class LSSGD(Optimizer):
    """
    Implements stochastic gradient descent with Laplacian smoothing (optionally with momentum).
    Nesterov momentum is based on the formula from:
        'On the importance of initialization and momentum in deep learning.'
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        sigma (float, optional): Laplacian smoothing parameter.
        momentum (float, optional): momentum factor (default: 0).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        dampening (float, optional): dampening for momentum (default: 0).
        nesterov (bool, optional): enables Nesterov momentum (default: False).
    Example:
        >>> optimizer = Grad_SGD(model.parameters(), lr=0.1, sigma=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    Considering the specific case of Momentum, the update can be written as:
        v = rho * v + g
        p = p - lr * v
    where p, g, v and rho denote the parameters, gradient, velocity, and momentum respectively.
    """
    def __init__(self, params, lr=required, sigma=0.05, eps = 0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, sigma=sigma, eps=eps, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LSSGD, self).__init__(params, defaults)
        
        sizes = []
        for param in self.param_groups[0]['params']:
            sizes.append(torch.numel(param))
        
        coeffs = []
        zero_Ns = []
        for size in sizes:
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
            coeffs.append(coeff)
            zero_Ns.append(zero_N)
        
        self.sigma = sigma
        self.eps = eps
        self.sizes = sizes
        self.coeffs = coeffs
        self.zero_Ns = zero_Ns
    
    
    def __setstate__(self, state):
        super(LSSGD, self).__setstate__(state)
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
        
        #import ipdb; ipdb.set_trace()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            # Update the parameters
            idx = 0
            for param in group['params']:
              #if idx <1:
                if param.grad is None:
                    continue
                tmp = param.grad.view(-1, self.sizes[idx])
                
                # Clip the gradient
                C = 1.0 #1.0
                tmp_norm2 = tmp.norm().data[0]#.numpy()[0]
                
                tmp = tmp/max(1., tmp_norm2/C)
                
                #eps = 0.01
                # Fix random seed for objective perturbation
                #if torch.cuda.is_available():
                #    torch.cuda.manual_seed_all(2019)
                #torch.manual_seed(2019)
                
                # Gaussian noise
                tmp = tmp.data + self.eps*torch.randn(tmp.shape).cuda() #Can make add noise part much faster, ones*random
                
                # Laplace noise
                #tmp = tmp.data + self.eps*torch.from_numpy((np.random.laplace(0, 1, tmp.shape).astype('float32'))).cuda()
                
                # Xiyang: no need to move to cpu
                #tmp = tmp.data

                #print(p.grad)
                #re, im = fft.fft(tmp, self.zero_Ns[idx])
                #re = re*self.coeffs[idx]
                #im = im*self.coeffs[idx]
                #tmp = fft.ifft(re, im)[0]
                
                '''
                ft_tmp = torch.rfft(tmp,1,onesided = False)
                tmp = torch.mul(ft_tmp,self.coeffs[idx])
                tmp = torch.irfft(tmp,1,onesided = False)
                '''
                
                ft_tmp = torch.rfft(tmp, 1, onesided=False)
                tmp = torch.zeros_like(ft_tmp)
                tmp[...,0] = ft_tmp[...,0]*self.coeffs[idx]
                tmp[...,1] = ft_tmp[...,1]*self.coeffs[idx]
                tmp = torch.irfft(tmp, 1, onesided = False)
                
                # tmp = Variable(tmp)  # Xiyang : I think this slows it down a little.
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

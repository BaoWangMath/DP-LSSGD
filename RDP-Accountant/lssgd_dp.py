import numpy as np
import torch
import random
import copy
from torch.autograd import Variable
from utils import progress_bar

def lssgd_dp(net, optimizer, train_loader, test_loader, loss_function, inner_iter_num, args, noise_multiplier, sigma=0.5):
    """
    Function to updated weights with a DPSGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss
    """
    train_loss = 0
    correct = 0
    total = 0
    # Run over the train_loader
    for batch_id, batch_data in enumerate(train_loader):
        if batch_id > inner_iter_num - 1:
            break
        
        # get the input and label
        inputs, labels = batch_data
        # wrap data and target into variable
        inputs = inputs.cuda()
        labels = labels.cuda()
        # calculate batch gradient and current loss
        beta=1                    
        #calculate per example gradient
        batch_grad, _=net.grad_dp_lay(inputs,labels,loss_function,beta)               
        #take SGD step
        optimizer.zero_grad()
        iter_grad=0
        for p_net in net.parameters():
            size_param = torch.numel(p_net)
            tmp = p_net.grad.view(-1, size_param)
            noise_g = beta*noise_multiplier*torch.zeros_like(tmp.data).normal_()
            tmp = batch_grad[iter_grad].view(-1, size_param)/args.BATCH_SIZE + noise_g/args.BATCH_SIZE
            
            # Perform Laplacian smoothing here!
            c = np.zeros(shape=(1, size_param))
            c[0, 0] = -2.; c[0, 1] = 1.; c[0, -1] = 1.
            c = torch.Tensor(c).cuda()
            zero_N = torch.zeros(1, size_param).cuda()
            c_fft = torch.rfft(c, 1, onesided=False)
            coeff = 1./(1.-sigma*c_fft[...,0])
            ft_tmp = torch.rfft(tmp, 1, onesided=False)
            tmp = torch.zeros_like(ft_tmp)
            tmp[...,0] = ft_tmp[...,0]*coeff
            tmp[...,1] = ft_tmp[...,1]*coeff
            tmp = torch.irfft(tmp, 1, onesided=False)
            tmp = tmp.view(p_net.grad.size())
            p_net.grad.data = tmp
            
            #noise_g = beta*noise_multiplier*torch.zeros_like(p_net.grad.data).normal_() 
            #p_net.grad.data = batch_grad[iter_grad]/args.BATCH_SIZE+noise_g/args.BATCH_SIZE
            
            iter_grad=iter_grad+1         
        optimizer.step()
        
        # Compute training error 
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.cuda().data).cpu().sum().item()
        
        if batch_id % 20 ==0:
            progress_bar(batch_id, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_id+1), 100.0/total*(correct), correct, total))
        
    train_loss=1 - correct*1./total
    
    net.eval() # Testing
    
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = loss_function(outputs, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
    
    test_loss=1 - correct*1./total
    
    return train_loss, test_loss

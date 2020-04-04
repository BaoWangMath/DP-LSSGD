import copy
import numpy as np
from torch.autograd import Variable
from utils import progress_bar
import torch
import torch.nn as nn
import time

def svrg_dp1(net, optimizer, train_loader, test_loader, loss_function, inner_iter_num, args, noise_multiplier):
    """
    Function to updated weights with a DPSVRG backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss
    """
    # record previous net full gradient
    pre_net_full = copy.deepcopy(net)
    # record previous net mini batch gradient
    pre_net_mini = copy.deepcopy(net)

    
    large_batch_num = args.LARGE_BATCH_NUMBER
    batch_size=args.BATCH_SIZE
    
    loss_function1=nn.CrossEntropyLoss(reduction='none')

    #Compute full grad
    beta=1
    pre_net_full.zero_grad()
    pre_net_full.grad_full_cal(train_loader, loss_function1, beta, large_batch_num)
        
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

        # compute current stochastic gradient
        optimizer.zero_grad()
        net.grad_mini_cal(inputs,labels,loss_function1,beta)
        
        # compute previous stochastic gradient
        pre_net_mini.zero_grad()
        pre_net_mini.grad_mini_cal(inputs,labels,loss_function1,beta)
        # take DPSVRG step
        for p_net, p_pre, p_full in zip(net.parameters(), pre_net_mini.parameters(), pre_net_full.parameters()):
            noise_g1 = 0*beta*noise_multiplier*torch.zeros_like(p_net.grad.data).normal_() 
            noise_g2 = 0*beta*noise_multiplier*torch.zeros_like(p_pre.grad.data).normal_() 
            noise_g3 = 0*beta*noise_multiplier*torch.zeros_like(p_full.grad.data).normal_() 
            p_net.grad.data =p_net.grad.data
            #p_net.grad.data =p_net.grad.data/batch_size+noise_g1/batch_size+ p_full.grad.data/batch_size * (1.0 /                                    large_batch_num)+noise_g3/batch_size* (1.0/large_batch_num)- p_pre.grad.data/batch_size-noise_g2/batch_size
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


    train_loss=1 - correct/total

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
            
    test_loss=1 - correct/total

    return train_loss, test_loss




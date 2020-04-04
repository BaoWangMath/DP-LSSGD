import copy
import numpy as np
from torch.autograd import Variable
from utils import progress_bar
import torch
import torch.nn as nn
import time

def svrg_dp(net, optimizer, train_loader, test_loader, loss_function, inner_iter_num, args, noise_multiplier):
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
    full_grad=pre_net_full.grad_dp_lay_full(train_loader,loss_function,beta,large_batch_num) 
    for iter_grad in range(len(full_grad)):
        noise_g = beta*noise_multiplier*torch.zeros_like(full_grad[iter_grad]).normal_() 
        full_grad[iter_grad]=full_grad[iter_grad]+noise_g
        
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
        ###
        
        # compute current stochastic gradient
        batch_grad, _=net.grad_dp_lay(inputs,labels,loss_function,beta) 
        # compute previous stochastic gradient
        pre_grad, _=pre_net_mini.grad_dp_lay(inputs,labels,loss_function,beta) 
        # take DPSVRG step
        optimizer.zero_grad()
        iter_grad=0
        for p_net in net.parameters():
            noise_g1 = beta*noise_multiplier*torch.zeros_like(p_net.grad.data).normal_() 
            noise_g2 = beta*noise_multiplier*torch.zeros_like(p_net.grad.data).normal_() 
            p_net.grad.data =batch_grad[iter_grad]/batch_size+noise_g1/batch_size+ full_grad[iter_grad]/batch_size * (1.0 /                                    large_batch_num)- pre_grad[iter_grad]/batch_size-noise_g2/batch_size
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




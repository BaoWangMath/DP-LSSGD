import copy
from torch.autograd import Variable
import torch
from utils import progress_bar


def sgd_dp(net, optimizer, train_loader, test_loader, loss_function, inner_iter_num, args, noise_multiplier):
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
            noise_g = beta*noise_multiplier*torch.zeros_like(p_net.grad.data).normal_() 
            p_net.grad.data = batch_grad[iter_grad]/args.BATCH_SIZE+noise_g/args.BATCH_SIZE 
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

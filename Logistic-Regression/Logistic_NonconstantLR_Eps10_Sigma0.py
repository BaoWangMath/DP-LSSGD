# -*- coding: utf-8 -*-
"""
SVM and Logistic regression
Logistic regression: cross-entropy loss;
SVM: hinge loss.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

from utils import *
from LS_SGD import *

parser = argparse.ArgumentParser(description='Convex ML (Logistic Regression or SVM)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=2.0, type=float, metavar='LR', help='initial learning rate')
#parser.add_argument('--eps', '--noise-level', default=0.1162, type=float, metavar='EPS', help='initial noise level') # 0.3
#parser.add_argument('--eps', '--noise-level', default=0.1393, type=float, metavar='EPS', help='initial noise level') # 0.25
#parser.add_argument('--eps', '--noise-level', default=0.1739, type=float, metavar='EPS', help='initial noise level') # 0.2
#parser.add_argument('--eps', '--noise-level', default=0.2316, type=float, metavar='EPS', help='initial noise level') # 0.15
parser.add_argument('--eps', '--noise-level', default=0.3471, type=float, metavar='EPS', help='initial noise level') # 0.1

parser.add_argument('--sigma', '--smoothing-const', default=0, type=float, metavar='SIGMA', help='initial smoothing const')



class Linear(nn.Module):
    """
    Linear model
    """
    def __init__(self):
        super(Linear, self).__init__()
        self.fc = nn.Linear(784, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        h = self.fc(x)
        return h

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    global best_acc
    best_acc = 0
    start_epoch = 0
    args = parser.parse_args()
    
    #--------------------------------------------------------------------------
    # Load the MNIST data
    #--------------------------------------------------------------------------
    #print('==> Preparing data...')
    root = './data'
    download = True
    
    trans=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=download)
    
    # Convert the data into appropriate torch format.
    kwargs = {'num_workers':1, 'pin_memory':True}
    
    batchsize_test = len(test_set)/2
    #print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    
    batchsize_train = 128
    #print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs
                                              )
    
    #--------------------------------------------------------------------------
    # Build the model
    #--------------------------------------------------------------------------
    net = Linear().cuda()
    
    # Print the model's information
    paramsList = list(net.parameters())
    kk = 0
    #for ii in paramsList:
    #    l = 1
    #    print('The structure of this layer: ' + str(list(ii.size())))
    #    for jj in ii.size():
    #        l *= jj
    #    print('The number of parameters in this layer: ' + str(l))
    #    kk = kk+l
    #print('Total number of parameters: ' + str(kk))
    
    #eps = 0.1 # Noise level, TO TUNE
    #sigma = 1. # LSGD parameter, 0 corresponding to SGD. TO TUNE
    #lr = 0.1 # Learning rate, we can tune this parameter, TO TUNE
    eps = args.eps
    sigma = args.sigma
    lr0 = args.lr # May
    weight_decay = 1e-4;
    
    nepoch = 50#200
    optimizer = LSSGD(net.parameters(), lr=lr0, sigma = sigma, eps = eps, momentum=0.0, weight_decay=weight_decay, nesterov=False) # May
    
    ###########################################################################
    # Logistic regression, constant step size
    ###########################################################################
    #lr = 0.1 (Laplace noise)
    # sigma    0              1              2       3         4           5
    #eps: 0   
    #   0.05  
    #   0.1   
    #   0.15
    #   0.2
    #   0.25
    #   0.3
    #   0.35
    #   0.4
    #   0.45
    #   0.5
    
    #loss1 = nn.MultiMarginLoss() # SVM
    loss1 = nn.CrossEntropyLoss() # Logistic Regression
    iter_count = 1
    
    for epoch in xrange(nepoch):
        #print('Epoch ID: ', epoch)
        
        #----------------------------------------------------------------------
        # Training constant step size
        #----------------------------------------------------------------------
        # Exponential decaying step size
        #if epoch == 50:
        #   lr = lr/10.
        lr = lr0/iter_count
        optimizer = LSSGD(net.parameters(), lr=lr, sigma = sigma, eps=eps, momentum=0.0, weight_decay=weight_decay, nesterov=False)
        correct = 0; total = 0; train_loss = 0
        net.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            score = net(x)
            loss = loss1(score, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data[0]
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            
            #progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #----------------------------------------------------------------------
        # Testing
        #----------------------------------------------------------------------
        test_loss = 0; correct = 0; total = 0
        net.eval()
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            score = net(x)
            loss = loss1(score, target)
            
            test_loss += loss.data[0]
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().numpy().sum() # May
            #progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc
        print epoch, correct/10000.
    #print('The best acc and 100th acc: ', best_acc, acc)

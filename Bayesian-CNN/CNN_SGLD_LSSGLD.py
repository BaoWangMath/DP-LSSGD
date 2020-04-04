# -*- coding: utf-8 -*-
"""
CNN: SGLD, LSSGLD (sigma=0.5, 1.0)
"""
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

clear_all()

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from os import path
import pickle
import copy

from Compute_Vec import Compute_Vec

from SGLD import SGLD
from LSSGLD import LSSGLD
from pSGLD import pSGLD
from LSpSGLD import LSpSGLD

#------------------------------------------------------------------------------
# Load the data
#------------------------------------------------------------------------------
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

batchsize_test = len(test_set)/10
test_dataloader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batchsize_test,
                                          shuffle=False, **kwargs
                                         )

batchsize_train = 100
train_dataloader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batchsize_train,
                                           shuffle=True, **kwargs
                                          )

train_dataloader2 = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=len(train_set)/10,
                                           shuffle=True, **kwargs
                                          )

#------------------------------------------------------------------------------
# Define the model
#------------------------------------------------------------------------------
class LeNet(nn.Module):
    """
    Wider LeNet
    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv1_drop = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(20, 20, 5, 1)
        self.fc = nn.Linear(4*4*20, 128)
        self.linear = nn.Linear(128, 10)
        self.loss = nn.CrossEntropyLoss()
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        
        # Flatten the tensor
        x = x.view(-1, 4*4*20)
        x = self.fc(x)
        x = self.linear(x)
        return x

nepoch = 50#100
burn_in = 3000 # Burn in iterataions
weight_decay = 5e-4
loss1 = nn.CrossEntropyLoss()

#------------------------------------------------------------------------------
# SGLD
#------------------------------------------------------------------------------
net = LeNet().cuda()
net2 = LeNet().cuda()

count = 0
for param in net.parameters():
    count += 1
    if count == 1:
        param1_data = copy.deepcopy(param.data)
    elif count == 2:
        param2_data = copy.deepcopy(param.data)
    elif count == 3:
        param3_data = copy.deepcopy(param.data)
    elif count == 4:
        param4_data = copy.deepcopy(param.data)
    elif count == 5:
        param5_data = copy.deepcopy(param.data)
    elif count == 6:
        param6_data = copy.deepcopy(param.data)
    elif count == 7:
        param7_data = copy.deepcopy(param.data)
    elif count == 8:
        param8_data = copy.deepcopy(param.data)
    else:
        print "There are more parameters in this model!"

param1_data -= param1_data
param2_data -= param2_data
param3_data -= param3_data
param4_data -= param4_data
param5_data -= param5_data
param6_data -= param6_data
param7_data -= param7_data
param8_data -= param8_data

train_loss_SGLD = []
test_loss_SGLD = []
train_acc_SGLD = []
test_acc_SGLD = []


iter_count = 0
for _ in range(nepoch):
    for x, target in train_dataloader:
        iter_count += 1
        net.train()
        optimizer = SGLD(net.parameters(), lr=0.01, momentum=0., weight_decay=weight_decay, nesterov=False)
        optimizer.zero_grad()
        x, target = Variable(x.cuda()), Variable(target.cuda())
        score = net(x)
        loss = loss1(score, target)
        loss.backward()
        optimizer.step()
        
        if iter_count >= burn_in:
            count = 0
            for param in net.parameters():
                count += 1
                if iter_count == burn_in:
                    if count == 1:
                        param1_data = param.data
                    elif count == 2:
                        param2_data = param.data
                    elif count == 3:
                        param3_data = param.data
                    elif count == 4:
                        param4_data = param.data
                    elif count == 5:
                        param5_data = param.data
                    elif count == 6:
                        param6_data = param.data
                    elif count == 7:
                        param7_data = param.data
                    elif count == 8:
                        param8_data = param.data
                else:
                    if count == 1:
                        param1_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param1_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 2:
                        param2_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param2_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 3:
                        param3_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param3_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 4:
                        param4_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param4_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 5:
                        param5_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param5_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 6:
                        param6_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param6_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 7:
                        param7_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param7_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 8:
                        param8_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param8_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
            
            # Initialize net2 and apply it on both training and test set
            count = 0
            for param in net2.parameters():
                count += 1
                if count == 1:
                    param.data = param1_data
                elif count == 2:
                    param.data = param2_data
                elif count == 3:
                    param.data = param3_data
                elif count == 4:
                    param.data = param4_data
                elif count == 5:
                    param.data = param5_data
                elif count == 6:
                    param.data = param6_data
                elif count == 7:
                    param.data = param7_data
                elif count == 8:
                    param.data = param8_data
            
            if iter_count % 5 == 0 and iter_count >= burn_in:
                net2.eval()
                # Performance on the training set
                loss_sum_train = 0.
                correct_count_train = 0
                for x_whole, y_whole in train_dataloader2:
                    x_whole, y_whole = Variable(x_whole.cuda()), Variable(y_whole.cuda())
                    score = net2(x_whole)
                    loss = loss1(score, y_whole)
                    #loss_sum_train += loss.data[0]
                    loss_sum_train += loss.data.cpu().numpy()
                    _, predicted = torch.max(score.data, 1)
                    correct = predicted.eq(y_whole.data).cpu().sum()
                    correct_count_train += correct
                train_acc = correct_count_train.numpy()*1./60000
        
                # Performance on the test set
                loss_sum_test = 0.
                correct_count_test = 0
                for x_whole, y_whole in test_dataloader:
                    x_whole, y_whole = Variable(x_whole.cuda()), Variable(y_whole.cuda())
                    score = net2(x_whole)
                    loss = loss1(score, y_whole)
                    #loss_sum_test += loss.data[0]
                    loss_sum_test += loss.data.cpu().numpy()
                    _, predicted = torch.max(score.data, 1)
                    correct = predicted.eq(y_whole.data).cpu().sum()
                    correct_count_test += correct
                test_acc = correct_count_test.numpy()*1./10000
                
                print 'Train acc, train loss, test acc, test loss: ', train_acc, loss_sum_train, test_acc, loss_sum_test
                train_loss_SGLD.append(loss_sum_train)
                test_loss_SGLD.append(loss_sum_test)
                train_acc_SGLD.append(train_acc)
                test_acc_SGLD.append(test_acc)

np.savetxt('train_loss_SGLD.csv', train_loss_SGLD, delimiter=',')
np.savetxt('train_acc_SGLD.csv', train_acc_SGLD, delimiter=',')
np.savetxt('test_loss_SGLD.csv', test_loss_SGLD, delimiter=',')
np.savetxt('test_acc_SGLD.csv', test_acc_SGLD, delimiter=',')


#------------------------------------------------------------------------------
# LSSGLD: sigma = 0.5
#------------------------------------------------------------------------------
net = LeNet().cuda()
net2 = LeNet().cuda()
sigma = 0.5

# Compute the vector for sqrt of the Laplacian smoothing
if path.exists('output_list_sigma05.pkl'):
    with open('output_list_sigma05.pkl', 'rb') as fp:
        vecs = pickle.load(fp)
else:
    vecs = []
    for param in net.parameters():
        len_param = torch.numel(param)
        print 'Len of the parameter: ', len_param
        if len_param > 2:
            conv_vec = Compute_Vec(len_param, sigma)
            vecs.append(conv_vec)
    with open('output_list_sigma05.pkl', 'wb') as fp:
        pickle.dump(vecs, fp)

vecs2 = []
for item in vecs:
    item2 = item.reshape((1, len(item)))
    vecs2.append(item2)
vecs = vecs2

count = 0
for param in net.parameters():
    count += 1
    if count == 1:
        param1_data = copy.deepcopy(param.data)
    elif count == 2:
        param2_data = copy.deepcopy(param.data)
    elif count == 3:
        param3_data = copy.deepcopy(param.data)
    elif count == 4:
        param4_data = copy.deepcopy(param.data)
    elif count == 5:
        param5_data = copy.deepcopy(param.data)
    elif count == 6:
        param6_data = copy.deepcopy(param.data)
    elif count == 7:
        param7_data = copy.deepcopy(param.data)
    elif count == 8:
        param8_data = copy.deepcopy(param.data)
    else:
        print "There are more parameters in this model!"

param1_data -= param1_data
param2_data -= param2_data
param3_data -= param3_data
param4_data -= param4_data
param5_data -= param5_data
param6_data -= param6_data
param7_data -= param7_data
param8_data -= param8_data

train_loss_LSSGLD_sigma05 = []
test_loss_LSSGLD_sigma05 = []
train_acc_LSSGLD_sigma05 = []
test_acc_LSSGLD_sigma05 = []

iter_count = 0
for _ in range(nepoch):
    for x, target in train_dataloader:
        iter_count += 1
        net.train()
        optimizer = LSSGLD(net.parameters(), lr=0.013, momentum=0., weight_decay=weight_decay, sigma=sigma, vecs=vecs)
        optimizer.zero_grad()
        x, target = Variable(x.cuda()), Variable(target.cuda())
        score = net(x)
        loss = loss1(score, target)
        loss.backward()
        optimizer.step()
        
        if iter_count >= burn_in:
            count = 0
            for param in net.parameters():
                count += 1
                if iter_count == burn_in:
                    if count == 1:
                        param1_data = param.data
                    elif count == 2:
                        param2_data = param.data
                    elif count == 3:
                        param3_data = param.data
                    elif count == 4:
                        param4_data = param.data
                    elif count == 5:
                        param5_data = param.data
                    elif count == 6:
                        param6_data = param.data
                    elif count == 7:
                        param7_data = param.data
                    elif count == 8:
                        param8_data = param.data
                else:
                    if count == 1:
                        param1_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param1_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 2:
                        param2_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param2_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 3:
                        param3_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param3_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 4:
                        param4_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param4_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 5:
                        param5_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param5_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 6:
                        param6_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param6_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 7:
                        param7_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param7_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 8:
                        param8_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param8_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
            
            # Initialize net2 and apply it on both training and test set
            count = 0
            for param in net2.parameters():
                count += 1
                if count == 1:
                    param.data = param1_data
                elif count == 2:
                    param.data = param2_data
                elif count == 3:
                    param.data = param3_data
                elif count == 4:
                    param.data = param4_data
                elif count == 5:
                    param.data = param5_data
                elif count == 6:
                    param.data = param6_data
                elif count == 7:
                    param.data = param7_data
                elif count == 8:
                    param.data = param8_data
            
            if iter_count % 5 == 0 and iter_count >= burn_in:
                net2.eval()
                # Performance on the training set
                loss_sum_train = 0.
                correct_count_train = 0
                for x_whole, y_whole in train_dataloader2:
                    x_whole, y_whole = Variable(x_whole.cuda()), Variable(y_whole.cuda())
                    score = net2(x_whole)
                    loss = loss1(score, y_whole)
                    #loss_sum_train += loss.data[0]
                    loss_sum_train += loss.data.cpu().numpy()
                    _, predicted = torch.max(score.data, 1)
                    correct = predicted.eq(y_whole.data).cpu().sum()
                    correct_count_train += correct
                train_acc = correct_count_train.numpy()*1./60000
        
                # Performance on the test set
                loss_sum_test = 0.
                correct_count_test = 0
                for x_whole, y_whole in test_dataloader:
                    x_whole, y_whole = Variable(x_whole.cuda()), Variable(y_whole.cuda())
                    score = net2(x_whole)
                    loss = loss1(score, y_whole)
                    #loss_sum_test += loss.data[0]
                    loss_sum_test += loss.data.cpu().numpy()
                    _, predicted = torch.max(score.data, 1)
                    correct = predicted.eq(y_whole.data).cpu().sum()
                    correct_count_test += correct
                test_acc = correct_count_test.numpy()*1./10000
                
                print 'Train acc, train loss, test acc, test loss: ', train_acc, loss_sum_train, test_acc, loss_sum_test
                train_loss_LSSGLD_sigma05.append(loss_sum_train)
                test_loss_LSSGLD_sigma05.append(loss_sum_test)
                train_acc_LSSGLD_sigma05.append(train_acc)
                test_acc_LSSGLD_sigma05.append(test_acc)

np.savetxt('train_loss_LSSGLD_sigma05.csv', train_loss_LSSGLD_sigma05, delimiter=',')
np.savetxt('train_acc_LSSGLD_sigma05.csv', train_acc_LSSGLD_sigma05, delimiter=',')
np.savetxt('test_loss_LSSGLD_sigma05.csv', test_loss_LSSGLD_sigma05, delimiter=',')
np.savetxt('test_acc_LSSGLD_sigma05.csv', test_acc_LSSGLD_sigma05, delimiter=',')

#------------------------------------------------------------------------------
# LSSGLD: sigma = 1.
#------------------------------------------------------------------------------
net = LeNet().cuda()
net2 = LeNet().cuda()
sigma = 1.0

# Compute the vector for sqrt of the Laplacian smoothing
if path.exists('output_list_sigma10.pkl'):
    with open('output_list_sigma10.pkl', 'rb') as fp:
        vecs = pickle.load(fp)
else:
    vecs = []
    for param in net.parameters():
        len_param = torch.numel(param)
        print 'Len of the parameter: ', len_param
        if len_param > 2:
            conv_vec = Compute_Vec(len_param, sigma)
            vecs.append(conv_vec)
    with open('output_list_sigma10.pkl', 'wb') as fp:
        pickle.dump(vecs, fp)

vecs2 = []
for item in vecs:
    item2 = item.reshape((1, len(item)))
    vecs2.append(item2)
vecs = vecs2

count = 0
for param in net.parameters():
    count += 1
    if count == 1:
        param1_data = copy.deepcopy(param.data)
    elif count == 2:
        param2_data = copy.deepcopy(param.data)
    elif count == 3:
        param3_data = copy.deepcopy(param.data)
    elif count == 4:
        param4_data = copy.deepcopy(param.data)
    elif count == 5:
        param5_data = copy.deepcopy(param.data)
    elif count == 6:
        param6_data = copy.deepcopy(param.data)
    elif count == 7:
        param7_data = copy.deepcopy(param.data)
    elif count == 8:
        param8_data = copy.deepcopy(param.data)
    else:
        print "There are more parameters in this model!"

param1_data -= param1_data
param2_data -= param2_data
param3_data -= param3_data
param4_data -= param4_data
param5_data -= param5_data
param6_data -= param6_data
param7_data -= param7_data
param8_data -= param8_data

train_loss_LSSGLD_sigma10 = []
test_loss_LSSGLD_sigma10 = []
train_acc_LSSGLD_sigma10 = []
test_acc_LSSGLD_sigma10 = []

iter_count = 0
for _ in range(nepoch):
    for x, target in train_dataloader:
        iter_count += 1
        net.train()
        optimizer = LSSGLD(net.parameters(), lr=0.015, momentum=0., weight_decay=weight_decay, sigma=sigma, vecs=vecs)
        optimizer.zero_grad()
        x, target = Variable(x.cuda()), Variable(target.cuda())
        score = net(x)
        loss = loss1(score, target)
        loss.backward()
        optimizer.step()
        
        if iter_count >= burn_in:
            count = 0
            for param in net.parameters():
                count += 1
                if iter_count == burn_in:
                    if count == 1:
                        param1_data = param.data
                    elif count == 2:
                        param2_data = param.data
                    elif count == 3:
                        param3_data = param.data
                    elif count == 4:
                        param4_data = param.data
                    elif count == 5:
                        param5_data = param.data
                    elif count == 6:
                        param6_data = param.data
                    elif count == 7:
                        param7_data = param.data
                    elif count == 8:
                        param8_data = param.data
                else:
                    if count == 1:
                        param1_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param1_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 2:
                        param2_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param2_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 3:
                        param3_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param3_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 4:
                        param4_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param4_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 5:
                        param5_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param5_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 6:
                        param6_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param6_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 7:
                        param7_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param7_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
                    elif count == 8:
                        param8_data = (iter_count*1. - burn_in*1.)/(iter_count*1. - burn_in*1. + 1.)*param8_data + 1./(iter_count*1. - burn_in*1. + 1.)*param.data
            
            # Initialize net2 and apply it on both training and test set
            count = 0
            for param in net2.parameters():
                count += 1
                if count == 1:
                    param.data = param1_data
                elif count == 2:
                    param.data = param2_data
                elif count == 3:
                    param.data = param3_data
                elif count == 4:
                    param.data = param4_data
                elif count == 5:
                    param.data = param5_data
                elif count == 6:
                    param.data = param6_data
                elif count == 7:
                    param.data = param7_data
                elif count == 8:
                    param.data = param8_data
            
            if iter_count % 5 == 0 and iter_count >= burn_in:
                net2.eval()
                # Performance on the training set
                loss_sum_train = 0.
                correct_count_train = 0
                for x_whole, y_whole in train_dataloader2:
                    x_whole, y_whole = Variable(x_whole.cuda()), Variable(y_whole.cuda())
                    score = net2(x_whole)
                    loss = loss1(score, y_whole)
                    #loss_sum_train += loss.data[0]
                    loss_sum_train += loss.data.cpu().numpy()
                    _, predicted = torch.max(score.data, 1)
                    correct = predicted.eq(y_whole.data).cpu().sum()
                    correct_count_train += correct
                train_acc = correct_count_train.numpy()*1./60000
        
                # Performance on the test set
                loss_sum_test = 0.
                correct_count_test = 0
                for x_whole, y_whole in test_dataloader:
                    x_whole, y_whole = Variable(x_whole.cuda()), Variable(y_whole.cuda())
                    score = net2(x_whole)
                    loss = loss1(score, y_whole)
                    #loss_sum_test += loss.data[0]
                    loss_sum_test += loss.data.cpu().numpy()
                    _, predicted = torch.max(score.data, 1)
                    correct = predicted.eq(y_whole.data).cpu().sum()
                    correct_count_test += correct
                test_acc = correct_count_test.numpy()*1./10000
                
                print 'Train acc, train loss, test acc, test loss: ', train_acc, loss_sum_train, test_acc, loss_sum_test
                train_loss_LSSGLD_sigma10.append(loss_sum_train)
                test_loss_LSSGLD_sigma10.append(loss_sum_test)
                train_acc_LSSGLD_sigma10.append(train_acc)
                test_acc_LSSGLD_sigma10.append(test_acc)

np.savetxt('train_loss_LSSGLD_sigma10.csv', train_loss_LSSGLD_sigma10, delimiter=',')
np.savetxt('train_acc_LSSGLD_sigma10.csv', train_acc_LSSGLD_sigma10, delimiter=',')
np.savetxt('test_loss_LSSGLD_sigma10.csv', test_loss_LSSGLD_sigma10, delimiter=',')
np.savetxt('test_acc_LSSGLD_sigma10.csv', test_acc_LSSGLD_sigma10, delimiter=',')

#------------------------------------------------------------------------------
# Plot the results
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt
ax = plt.subplot(111, xlabel='x', ylabel='y', title='title')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)
plt.figure(1, figsize=(7, 6))
plt.clf()
x = list(range(burn_in, burn_in + len(test_loss_SGLD)))
plt.plot(x, test_loss_SGLD, 'r', lw=1, label='SGLD')
plt.plot(x, test_loss_LSSGLD_sigma05, 'b', lw=1, label='LSSGLD (\sigma = 0.5)')
plt.plot(x, test_loss_LSSGLD_sigma10, 'k', lw=1, label='LSSGLD (\sigma = 1.0)')
plt.legend()
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Test error')
plt.show()

# -*- coding: utf-8 -*-
"""
Bayesian logistic regression
LS SGLD pima
"""
'''
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
        

clear_all()
'''

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import copy

from Compute_Vec import *

from LS_SGLD import *
from LSSGD2 import *


pima = np.genfromtxt('pima-indians-diabetes.data', delimiter=',')
names = ["Number of times pregnant",
         "Plasma glucose concentration",
         "Diastolic blood pressure (mm Hg)",
         "Triceps skin fold thickness (mm)",
         "2-Hour serum insulin (mu U/ml)",
         "Body mass index (weight in kg/(height in m)^2)",
         "Diabetes pedigree function",
         "Age (years)",
         "Class variable (0 or 1)"]

# Load data
X = np.concatenate((np.ones((pima.shape[0], 1)), pima[:, 0:8]), axis=1) # Features
Y = pima[:, 8] # Label

data_feature_pima = X
data_label_pima = Y.astype(int)

x_train, x_test, y_train, y_test = train_test_split(data_feature_pima, data_label_pima, test_size=0.2, random_state=85)
scaler = StandardScaler()
transformed = scaler.fit_transform(x_train)
train = data_utils.TensorDataset(torch.from_numpy(transformed).float(), torch.from_numpy(y_train))#.float())
batch_size = 5#5#10
train_dataloader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
num_batch = len(train)/batch_size
train_dataloader2 = data_utils.DataLoader(train, batch_size=len(train), shuffle=True)

scaler = StandardScaler()
transformed = scaler.fit_transform(x_test)
test = data_utils.TensorDataset(torch.from_numpy(transformed).float(), torch.from_numpy(y_test))#.float())
test_dataloader = data_utils.DataLoader(test, batch_size=len(test), shuffle=False)

ndim = data_feature_pima.shape[1]



'''
ijcnn = np.genfromtxt('ijcnn.csv', delimiter=',')

# Load data
X = np.concatenate((np.ones((ijcnn.shape[0], 1)), ijcnn[:, 0:22]), axis=1) # Features
Y = ijcnn[:, 22] # Label
Y[Y==-1] = 0

data_feature_ijcnn = X
data_label_ijcnn = Y.astype(int)

x_train, x_test, y_train, y_test = train_test_split(data_feature_ijcnn, data_label_ijcnn, test_size=0.2, random_state=85)
scaler = StandardScaler()
transformed = scaler.fit_transform(x_train)
train = data_utils.TensorDataset(torch.from_numpy(transformed).float(), torch.from_numpy(y_train))#.float())
batch_size = 2#10#5#10
train_dataloader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
num_batch = len(train)/batch_size
train_dataloader2 = data_utils.DataLoader(train, batch_size=len(train), shuffle=True)

scaler = StandardScaler()
transformed = scaler.fit_transform(x_test)
test = data_utils.TensorDataset(torch.from_numpy(transformed).float(), torch.from_numpy(y_test))#.float())
test_dataloader = data_utils.DataLoader(test, batch_size=len(test), shuffle=False)

ndim = data_feature_ijcnn.shape[1]
'''


# Define the model
class Linear(nn.Module):
    """
    Linear model
    """
    def __init__(self):
        super(Linear, self).__init__()
        self.fc = nn.Linear(ndim, 2)
    
    def forward(self, x):
        h = self.fc(x)
        return h


def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.)
        m.weight_bias.fill_(0.)

best_acc = 0.
best_cum = 0

#nepoch = 1#10 # batch size 2
nepoch = 10
#lr0 = 1e-2 # Good
lr0 = 1e-2*1.5
weight_decay = 1e-4

loss1 = nn.CrossEntropyLoss() # Logistic regression

#------------------------------------------------------------------------------
# Training with LS-SGLD
#------------------------------------------------------------------------------
net = Linear().cuda()
weights_init(net)

net2 = Linear().cuda()

vecs_len = []
vecs = []
sigma = 1.

for param in net.parameters():
    len_param = torch.numel(param)
    print 'Len of the parameter: ', len_param
    if len_param > 2:
        conv_vec = Compute_Vec(len_param, sigma)
        vecs.append(conv_vec)
    #else:
    #    vecs.append(np.zeros(shape=(1, 2)))

count = 0
for param in net.parameters():
    count += 1
    if count == 1:
        param1_data = copy.deepcopy(param.data)
    elif count == 2:
        param2_data = copy.deepcopy(param.data)

param1_data = param1_data - param1_data
param2_data = param2_data - param2_data

#burn_in = 50#10
burn_in = 10


#------------------------------------------------------------------------------
# LS SGLD
#------------------------------------------------------------------------------
train_loss_LS = []
test_loss_LS = []

train_correct_list_LS = []
test_correct_list_LS = []

count_iteration = 0

for _ in range(nepoch):
    lr = lr0
    train_correct_sum = 0
    test_correct_sum = 0
    for x, target in train_dataloader:
        net.train()
        count_iteration += 1
        if count_iteration < burn_in:
            # Use LS SGLD
            optimizer = LSSGLD2(net.parameters(), lr=lr, sigma=sigma, momentum=0., weight_decay=weight_decay, nesterov=False, vecs=vecs)
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            score = net(x)
            loss = loss1(score, target)
            print loss, loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()
        else:
            # Use SGLD
            optimizer = LS_SGLD(net.parameters(), lr=lr, sigma=0, momentum=0., weight_decay=weight_decay, nesterov=False)
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            score = net(x)
            loss = loss1(score, target)
            print loss, loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()
            
            count = 0
            for param in net.parameters():
                count += 1
                
                if count == 1:
                    param1_data = param.data
                    #param1_data = 1./(count_iteration-burn_in+1)*param.data + (count_iteration-burn_in)/(count_iteration-burn_in+1)*param1_data
                elif count == 2:
                    param2_data = param.data
                    #param2_data = 1./(count_iteration-burn_in+1)*param.data + (count_iteration-burn_in)/(count_iteration-burn_in+1)*param2_data
        train_loss_tmp = loss.data[0]
        _, predicted = torch.max(score.data, 1)
        correct_train = predicted.eq(target.data).cpu().sum()
        train_correct_sum += correct_train
        
        # Initialize net2
        count = 0
        for param in net2.parameters():
            count += 1
            if count == 1:
                param.data = param1_data
            elif count == 2:
                param.data = param2_data
        # Compute the training loss over the whole training set, we can do in a mini-batch fashion and sum them
        net2.eval()
        loss_sum = 0.
        for x_whole, y_whole in train_dataloader2:
            x_whole, y_whole = Variable(x_whole.cuda()), Variable(y_whole.cuda())
            score = net2(x_whole)
            loss = loss1(score, y_whole)
            loss_sum += loss.data[0]
        train_loss_LS.append(loss_sum/len(train_dataloader))
        
        # Test
        test_loss_tmp = 0
        correct_test = 0
        net.eval()
        for x_test, target_test in test_dataloader:
            x_test, target_test = Variable(x_test.cuda(), volatile=True), Variable(target_test.cuda(), volatile=True)
            score = net2(x_test)
            loss = loss1(score, target_test)
            test_loss_tmp += loss.data[0]
            _, predicted = torch.max(score.data, 1)
            correct_test += predicted.eq(target_test.data).cpu().sum()
        test_loss_LS.append(test_loss_tmp)
        test_correct_sum = correct_test
        
    train_correct_list_LS.append(train_correct_sum)
    test_correct_list_LS.append(test_correct_sum)


#------------------------------------------------------------------------------
# Training with SGLD
#------------------------------------------------------------------------------
net = Linear().cuda()
weights_init(net)

train_loss = []
test_loss = []

train_correct_list = []
test_correct_list = []

count_iteration = 0

for epoch_id in range(nepoch):
    # Decaying learning rate
    lr = lr0#0.05#lr0
    train_correct_sum = 0
    test_correct_sum = 0
    idx_count = 0
    for x, target in train_dataloader:
        count_iteration += 1
        net.train()
        optimizer = LS_SGLD(net.parameters(), lr=lr, sigma=0., momentum=0., weight_decay=weight_decay, nesterov=False)
        optimizer.zero_grad()
        x, target = Variable(x.cuda()), Variable(target.cuda())
        score = net(x)
        loss = loss1(score, target)
        print loss, loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()
        
        # Moving Average of the parameters
        if count_iteration > burn_in:
          count = 0
          for param in net.parameters():
            count += 1
            
            if count == 1:
                param1_data = param.data
                #param1_data = 1./(count_iteration-burn_in+1)*param.data + (count_iteration-burn_in)/(count_iteration-burn_in+1)*param1_data
            elif count == 2:
                param2_data = param.data
                #param2_data = 1./(count_iteration-burn_in+1)*param.data + (count_iteration-burn_in)/(count_iteration-burn_in+1)*param2_data
            '''
            if count_iteration == burn_in:
                if count == 1:
                    param1 = param.data
                elif count == 2:
                    param2 = param.data
            elif count_iteration > burn_in:
                if count == 1:
                    param.data = 1./(count_iteration-burn_in+1)*param.data + (count_iteration*1.-burn_in*1.)/(count_iteration-burn_in+1)*param1
                    param1 = param.data # Record the moving average
                elif count == 2:
                    param.data = 1./(count_iteration-burn_in+1)*param.data + (count_iteration*1.-burn_in*1.)/(count_iteration-burn_in+1)*param2
                    param2 = param.data # Record the moving average
            '''
        train_loss_tmp = loss.data[0]
        _, predicted = torch.max(score.data, 1)
        correct_train = predicted.eq(target.data).cpu().sum()
        train_correct_sum += correct_train
        ##train_loss.append(train_loss_tmp.cpu().numpy())
        #train_loss.append(train_loss_tmp)
        
        # Initialize net2
        count = 0
        for param in net2.parameters():
            count += 1
            if count == 1:
                param.data = param1_data
            elif count == 2:
                param.data = param2_data
        
        # Compute the training loss over the whole training set, we can do in a mini-batch fashion and sum them
        net.eval()
        loss_tmp = 0.
        for x_whole, y_whole in train_dataloader2:
            x_whole, y_whole = Variable(x_whole.cuda()), Variable(y_whole.cuda())
            score = net2(x_whole)
            loss = loss1(score, y_whole)
            loss_tmp += loss.data[0]
        train_loss.append(loss_tmp/len(train_dataloader))
        
        # Test
        test_loss_tmp = 0
        correct_test = 0
        
        for x_test, target_test in test_dataloader:
            x_test, target_test = Variable(x_test.cuda(), volatile=True), Variable(target_test.cuda(), volatile=True)
            score = net2(x_test)
            loss = loss1(score, target_test)
            test_loss_tmp += loss.data[0]
            _, predicted = torch.max(score.data, 1)
            correct_test += predicted.eq(target_test.data).cpu().sum()
        test_correct_sum = correct_test
        #test_loss.append(test_loss_tmp.cpu().numpy())
        test_loss.append(test_loss_tmp)
        
    train_correct_list.append(train_correct_sum)
    test_correct_list.append(test_correct_sum)


ax = plt.subplot(111, xlabel='x', ylabel='y', title='title')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)
plt.figure(1, figsize=(7, 6))
plt.clf()
plt.plot(train_loss_LS[burn_in:-1:1], 'g', lw=1, label='LS-SGLD')
plt.plot(train_loss[burn_in:-1:1], 'r', lw=1, label='SGLD')
plt.legend()
plt.grid()
plt.xlabel('iterations')
plt.ylabel('Training Loss')
plt.show()

ax = plt.subplot(111, xlabel='x', ylabel='y', title='title')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)
plt.figure(1, figsize=(7, 6))
plt.clf()
plt.plot(test_loss_LS[burn_in:-1:1], 'g', lw=1, label='LS-SGLD')
plt.plot(test_loss[burn_in:-1:1], 'r', lw=1, label='SGLD')
plt.legend()
plt.grid()
plt.xlabel('iterations')
plt.ylabel('Test Loss')
plt.show()

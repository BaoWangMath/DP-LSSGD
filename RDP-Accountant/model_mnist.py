import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import copy

class ConvNet(nn.Module):
    
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
        '''
        self.fc = nn.Linear(784, 10)
    
    def forward(self, x):
        '''
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
        '''
        x = x.view(-1, 28*28)
        h = self.fc(x)
        return h
    
    '''
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        #self.linear1 = nn.Linear(512, 32)
        #self.fc = nn.Linear(32, num_classes)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        #out = self.linear1(out)
        out = self.fc(out)
        return out
    '''
    
    def grad_dp_lay(self, inputs,labels,loss_function,beta):        
        outputs = self(inputs)
        #calculate per example gradient
        per_grad_list=[]
        for idx in range(len(inputs)):
            self.zero_grad()
            per_outputs=outputs[idx].view(-1,10)
            per_labels=labels[idx].view(-1)
            per_loss = loss_function(per_outputs, per_labels)
            per_loss.backward(retain_graph=True)
            per_grad=[]
            iter_num=0
            per_grad_norm=0           
            for param in self.parameters():
                per_grad.append(copy.deepcopy(param.grad.data))
                per_grad_norm = per_grad[iter_num].norm(2)
                clip_val=max(1,per_grad_norm/beta)
                per_grad[iter_num]=per_grad[iter_num]/clip_val
                iter_num+=1
            per_grad_list.append(per_grad)
        batch_grad=[]
        for param_num in range(len(per_grad_list[0])):
            temp_tenor=per_grad_list[0][param_num]
            for idx in range(len(per_grad_list)-1):
                temp_tenor=temp_tenor+per_grad_list[idx+1][param_num]
            batch_grad.append(temp_tenor)        
        return batch_grad, per_grad_norm
    
    def grad_dp_lay_full(self, dataset, loss_function, beta, large_batch_num): 
        for idx, (inputs, targets) in enumerate(dataset):
            if idx > large_batch_num - 1:
                break
            if idx == 0:
                inputs, targets = inputs.cuda(), targets.cuda()
                full_grad,_=self.grad_dp_lay(inputs,targets,loss_function,beta)
            if idx > 0:
                inputs, targets = inputs.cuda(), targets.cuda()
                batch_grad,_=self.grad_dp_lay(inputs,targets,loss_function,beta)
                full_grad=np.add(full_grad,batch_grad)
        return full_grad
    
    def grad_full_cal(self, dataset, loss_function, beta, large_batch_num):
        for idx, (inputs, targets) in enumerate(dataset):
            if idx > large_batch_num - 1:
                break
            inputs, targets = inputs.cuda(), targets.cuda()
            self.grad_mini_cal(inputs,targets,loss_function,beta)
            
    def grad_mini_cal(self,inputs,targets,loss_function,beta):
        outputs = self(inputs)
        loss = loss_function(outputs, targets)
        for idx in range(len(inputs)):
            loss[idx].backward(retain_graph=True)
            for param in self.parameters():
                per_grad_norm = param.grad.data.norm(2)
                clip_val=max(1,per_grad_norm/beta)
                param.grad.data=param.grad.data/clip_val

    
    
    

    


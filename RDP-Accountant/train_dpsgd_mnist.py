from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import argparse


from sgd_dp import *
from model_mnist import *

from rdp_accountant_nn import compute_rdp
from rdp_accountant_nn import get_privacy_spent


# Training settings
parser = argparse.ArgumentParser(description='FLASH Example')

parser.add_argument('--BATCH-SIZE', type=int, default=256, metavar='N',
                    help='mini batch size for scsg in training (default: 100)')
parser.add_argument('--LR-SGD', type=float, default=0.05, metavar='LR',
                    help='learning rate for scsg (default: 0.6)')
parser.add_argument('--EPOCH', type=int, default=500, metavar='LR',
                    help='total epoch (data pass) for the algorithm (default: 500)')
args = parser.parse_args()

# # MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_dataset = torchvision.datasets.MNIST(
    root='./data/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.BATCH_SIZE,
                                          shuffle=False)

def main():
    noise_multiplier=1.25
    def compute_epsilon(steps,l_num,noise_multiplier):
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        sampling_probability = args.BATCH_SIZE/60000
        rdp = compute_rdp(q=sampling_probability,
                      noise_multiplier=noise_multiplier,
                      steps=steps,
                      orders=orders,l_num=l_num)
        #Delta is set to 1e-5 because MNIST has 60000 training points.
        return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

    # define CNN net
    net = ConvNet(10).cuda()
    # define loss function
    loss_func=nn.CrossEntropyLoss()
    # setup optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=args.LR_SGD)
       
    # training loop
    real_epoch=0
    for epoch in range(args.EPOCH):
        inner_iter_num =10
        cur_train_loss, cur_test_loss= sgd_dp(net, optimizer, train_loader,
                                                                 test_loader, loss_func,
                                                                 inner_iter_num, args, noise_multiplier)
        real_epoch=real_epoch+inner_iter_num*args.BATCH_SIZE/60000
        # print progress
        print('Epoch: %.3f' % real_epoch,
              '| train error: %.3f' % cur_train_loss,
              '| test error: %.3f' % cur_test_loss)
        eps = compute_epsilon((epoch+1) * inner_iter_num,4,noise_multiplier)
        print('For delta=1e-5, the current epsilon is: %.2f' % eps)
if __name__ == '__main__':
    main()

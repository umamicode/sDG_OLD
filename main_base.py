
'''
train base model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

#PRISM
#[TODO]
from network.modules import ReLIC_Loss, get_resnet
from network.modules.transformations import TransformsRelic
from network.modules.sync_batchnorm import convert_model


import os
import click
import time
import numpy as np

from network import mnist_net, res_net
import data_loader

HOME = os.environ['HOME']

@click.command()
@click.option('--gpu', type=str, default='0', help='Choose GPU')
@click.option('--data', type=str, default='mnist', help='Dataset name')
@click.option('--ntr', type=int, default=None, help='Select the first ntr samples of the training set')
@click.option('--translate', type=float, default=None, help='Random translation data augmentation')
@click.option('--autoaug', type=str, default=None, help='AA FastAA RA')
@click.option('--epochs', type=int, default=100)
@click.option('--nbatch', type=int, default=None, help='The number of batches in each epoch')
@click.option('--batchsize', type=int, default=256, help='The number of samples in each batch')
@click.option('--lr', type=float, default=1e-3)
@click.option('--lr_scheduler', type=str, default='none', help='Learning Weight Decay')
@click.option('--svroot', type=str, default='./saved', help='Project file save path')
@click.option('--backbone', type=str, default= 'custom', help= 'Backbone Model (custom/resnet18,resnet50')

def experiment(gpu, data, ntr, translate, autoaug, epochs, nbatch, batchsize, lr, lr_scheduler, svroot, backbone):
    settings = locals().copy()
    print(settings)

    # Global Settings
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if not os.path.exists(svroot):
        os.makedirs(svroot)
    writer = SummaryWriter(svroot)
    
    # Load datasets and models
    if data in ['mnist', 'mnist_t']:
        # Load dataset
        if data == 'mnist':
            trset = data_loader.load_mnist('train', translate=translate, ntr=ntr, autoaug=autoaug)
        elif data == 'mnist_t':
            trset = data_loader.load_mnist_t('train', translate=translate, ntr=ntr)
        teset = data_loader.load_mnist('test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        
        if backbone == 'custom':
            cls_net = mnist_net.ConvNet().cuda()
            cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
        elif backbone in ['resnet18','resnet50']:
            encoder = get_resnet(backbone, pretrained= True) # Pretrained Backbone default as True
            n_features = encoder.fc.in_features
            output_dim= 10
            cls_net= res_net.ConvNet(encoder, 128, n_features, output_dim).cuda() #projection_dim/ n_features
            cls_opt = optim.Adam(cls_net.parameters(), lr=lr)

    elif data == 'mnistvis':
        trset = data_loader.load_mnist('train')
        teset = data_loader.load_mnist('test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        #[TODO] - add resnet option here
        cls_net= mnist_net.ConvNetVis().cuda() 
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
    
    elif data == 'cifar10':
        print("TODO- ResNet Not Implemented atm.")
        # Load Dataset
        trset = data_loader.load_cifar10(split='train', autoaug=autoaug)
        teset = data_loader.load_cifar10(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        #[TODO- WideResNet?- MNIST_NET is too shallow]
        #cls_net = wideresnet.WideResNet(16, 10, 4).cuda()
        cls_net= mnist_net.ConvNet().cuda()

        cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs)
    elif 'synthia' in data:
        # Load Dataset
        branch = data.split('_')[1]
        trset = data_loader.load_synthia(branch)
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True)
        teloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True)
        imsize = [192, 320]
        nclass = 14
        # Load Model
        cls_net = fcn.FCN_resnet50(nclass=nclass).cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)#, weight_decay=1e-4) 
        # For synthia: adding weight_decay will drop 1-2 points
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs*len(trloader))
    
    cls_criterion = nn.CrossEntropyLoss()

    # Train Start
    best_acc = 0
    for epoch in range(epochs):
        t1 = time.time()
        
        loss_list = []
        cls_net.train()
        for i, (x, y) in enumerate(trloader):
            x, y = x.cuda(), y.cuda()
            
            # Train
            p = cls_net(x)
            cls_loss = cls_criterion(p, y)
            #torch.cuda.synchronize()
            cls_opt.zero_grad()
            cls_loss.backward()
            cls_opt.step()
            
            loss_list.append([cls_loss.item()])
            
            # Adjust Learning Rate
            if lr_scheduler in ['cosine']:
                scheduler.step()

        cls_loss, = np.mean(loss_list, 0)
        

        # Test and Save Optimal Model
        cls_net.eval()
        if data in ['mnist', 'mnist_t', 'cifar10', 'mnistvis']:
            teacc = evaluate(cls_net, teloader)
        elif 'synthia' in data:
            teacc = evaluate_seg(cls_net, teloader, nclass) # Counting miou

        if best_acc < teacc:
            best_acc = teacc
            torch.save({'cls_net':cls_net.state_dict()}, os.path.join(svroot, 'best.pkl'))

        # Save Log
        t2 = time.time()
        print(f'epoch {epoch}, time {t2-t1:.2f}, cls_loss {cls_loss:.4f} teacc {teacc:2.2f}')
        writer.add_scalar('scalar/cls_loss', cls_loss, epoch)
        writer.add_scalar('scalar/teacc', teacc, epoch)

    writer.close()

def evaluate(net, teloader):
    correct, count = 0, 0
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(teloader):
        with torch.no_grad():
            x1 = x1.cuda()
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    # Calculate the evaluation index
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    return acc

if __name__=='__main__':
    experiment()


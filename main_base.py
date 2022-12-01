
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

import os
import click
import time
import numpy as np

from network import mnist_net, res_net
from network.modules import get_resnet
import data_loader

HOME = os.environ['HOME']
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'

@click.command()
@click.option('--gpu', type=str, default='0', help='Choose GPU')
@click.option('--data', type=str, default='mnist', help='Dataset name (mnist/cifar10/pacs')
@click.option('--ntr', type=int, default=None, help='Select the first ntr samples of the training set')
@click.option('--translate', type=float, default=None, help='Random translation data augmentation')
@click.option('--autoaug', type=str, default=None, help='AA FastAA RA')
@click.option('--epochs', type=int, default=100)
@click.option('--nbatch', type=int, default=None, help='The number of batches in each epoch')
@click.option('--batchsize', type=int, default=256, help='The number of samples in each batch')
@click.option('--lr', type=float, default=1e-3)
@click.option('--lr_scheduler', type=str, default='none', help='Learning Weight Decay')
@click.option('--svroot', type=str, default='./saved', help='Project file save path')
@click.option('--backbone', type=str, default= 'custom', help= 'Backbone Model (custom/resnet18,resnet50,wideresnet)')
@click.option('--pretrained', type=str, default= 'False', help= 'Pretrained Backbone - ResNet18/50, Custom MNISTnet does not matter')
@click.option('--projection_dim', type=int, default=128, help= "Projection Dimension of the representation vector for Resnet; Default: 128")

def experiment(gpu, data, ntr, translate, autoaug, epochs, nbatch, batchsize, lr, lr_scheduler, svroot, backbone, pretrained, projection_dim):
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
            cls_net = mnist_net.ConvNet(projection_dim=projection_dim).cuda()
            cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
        elif backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as True
            n_features = encoder.fc.in_features
            output_dim= 10
            cls_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() #projection_dim/ n_features
            cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
        
    elif data in ['mnistvis']:
        trset = data_loader.load_mnist('train')
        teset = data_loader.load_mnist('test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        #[TODO] - add resnet option here
        cls_net= mnist_net.ConvNetVis().cuda() 
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
    
    elif data in ['cifar10']:
        
        # Load Dataset
        trset = data_loader.load_cifar10(split='train')
        teset = data_loader.load_cifar10(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        
        if backbone == 'custom':
            cls_net = mnist_net.ConvNet(projection_dim=projection_dim).cuda()
            cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
        elif backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as True
            n_features = encoder.fc.in_features
            output_dim= 10
            cls_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() #projection_dim/ n_features
            cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
            #cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4) 
        
            ###### Old Code
            #[TODO- WideResNet?- MNIST_NET is too shallow]
            #cls_net = wideresnet.WideResNet(16, 10, 4).cuda()
            #cls_net= mnist_net.ConvNet().cuda()
            #cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs)

    elif data in ['pacs']:
        # Load Dataset
        trset = data_loader.load_pacs(split='train')
        teset = data_loader.load_pacs(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)

        if backbone == 'custom':
            raise ValueError('WORK IN PROGRESS: PLEASE USE Resnet-18/50 For PACS')
            #cls_net = mnist_net.ConvNet(projection_dim=projection_dim).cuda()
            #cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
        elif backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as True
            n_features = encoder.fc.in_features
            output_dim= 7
            cls_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() #projection_dim/ n_features
            cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
            #cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
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
        elif data in ['synthia']:
            teacc = evaluate_seg(cls_net, teloader, nclass) # Counting miou
        elif data in ['cifar10']:
            teacc = evaluate(cls_net, teloader)
        elif data in ['pacs']:
            teacc = evaluate(cls_net, teloader)
            
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


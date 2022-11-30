
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import numpy as np
import click
import pandas as pd

from network import mnist_net, res_net
#{TODO} Added ResNet
from network.modules import get_resnet

import data_loader
from main_base import evaluate
from utils import log

@click.command()
@click.option('--gpu', type=str, default='0', help='Select GPU number')
@click.option('--modelpath', type=str, default='saved/best.pkl')
@click.option('--svpath', type=str, default=None, help='Path to SaveLogs')
@click.option('--channels', type=int, default=3)
@click.option('--backbone', type=str, default= 'custom', help= 'Backbone Model (custom/resnet18,resnet50')
@click.option('--pretrained', type=str, default= 'False', help= 'Pretrained Backbone - ResNet18/50, Custom MNISTnet does not matter')
@click.option('--projection_dim', type=int, default=128, help= "Projection Dimension of the representation vector for Resnet; Default: 128")
@click.option('--data', type=str, default= 'mnist', help= 'Data to evaluate from (mnist/cifar10)')

def main(gpu, modelpath, svpath, backbone, channels, pretrained, projection_dim, data):
    print("--Testing Model from: {svroot}".format(svroot= modelpath))
    if data in ['mnist']:
        evaluate_digit(gpu, modelpath, svpath, backbone, pretrained,projection_dim, channels)
    elif data in ['cifar10']:
        evaluate_image(gpu, modelpath, svpath, backbone, pretrained,projection_dim, channels)
    elif data in ['pacs']:
        evaluate_pacs(gpu, modelpath, svpath, backbone, pretrained,projection_dim, channels)

def evaluate_digit(gpu, modelpath, svpath, backbone, pretrained,projection_dim, channels=3):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Load Model
    if backbone== 'custom':
        if channels == 3:
            cls_net = mnist_net.ConvNet(projection_dim).cuda()
        elif channels == 1:
            cls_net = mnist_net.ConvNet(projection_dim, imdim=channels).cuda()
    elif backbone in ['resnet18','resnet50','wideresnet']:
        if channels == 3:
            encoder = get_resnet(backbone, pretrained= pretrained)
            n_features = encoder.fc.in_features
            output_dim = 10 #{TODO}- output
            cls_net = res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda()
        elif channels == 1:
            encoder = get_resnet(backbone, pretrained= pretrained)
            n_features = encoder.fc.in_features
            output_dim = 10
            cls_net = res_net.ConvNet(encoder, projection_dim, n_features, output_dim, imdim=channels).cuda()

    saved_weight = torch.load(modelpath) #dict(saved_weight) only has cls_net as key
    cls_net.load_state_dict(saved_weight['cls_net'])
    #cls_net.eval()

    # Test
    str2fun = { 
        'mnist': data_loader.load_mnist,
        'mnist_m': data_loader.load_mnist_m,
        'usps': data_loader.load_usps,
        'svhn': data_loader.load_svhn,
        'syndigit': data_loader.load_syndigit,
        }   
    columns = ['mnist', 'mnist_m', 'usps', 'svhn', 'syndigit']
    rst = []
    for data in columns:
        teset = str2fun[data]('test', channels=channels)
        teloader = DataLoader(teset, batch_size=128, num_workers=8)
        teacc = evaluate(cls_net, teloader)
        rst.append(teacc)
    
    df = pd.DataFrame([rst], columns=columns)
    print(df)
    if svpath is not None:
        df.to_csv(svpath)

def evaluate_image(gpu, modelpath, svpath, backbone, pretrained,projection_dim, channels=3):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Load Model
    if backbone== 'custom':
        if channels == 3:
            cls_net = mnist_net.ConvNet(projection_dim).cuda()
        elif channels == 1:
            cls_net = mnist_net.ConvNet(projection_dim, imdim=channels).cuda()
    elif backbone in ['resnet18','resnet50','wideresnet']:
        if channels == 3:
            encoder = get_resnet(backbone, pretrained= pretrained)
            n_features = encoder.fc.in_features
            output_dim = 10 #{TODO}- output
            cls_net = res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda()
        elif channels == 1:
            encoder = get_resnet(backbone, pretrained= pretrained)
            n_features = encoder.fc.in_features
            output_dim = 10 #cifar10
            cls_net = res_net.ConvNet(encoder, projection_dim, n_features, output_dim, imdim=channels).cuda()

    saved_weight = torch.load(modelpath) #dict(saved_weight) only has cls_net as key
    cls_net.load_state_dict(saved_weight['cls_net'])
    #cls_net.eval()

    # Test
    str2fun = { 
        'cifar10': data_loader.load_cifar10,
        'cifar10c': data_loader.load_cifar10c,
        'cifar10_1': data_loader.load_cifar10_1,
        'stl10': data_loader.load_stl10,
        
        }   
    columns = ['cifar10','cifar10c','cifar10_1','stl10']
    rst = []
    for data in columns:
        teset = str2fun[data]('test', channels=channels)
        teloader = DataLoader(teset, batch_size=128, num_workers=8)
        teacc = evaluate(cls_net, teloader)
        rst.append(teacc)
    
    df = pd.DataFrame([rst], columns=columns)
    print(df)
    if svpath is not None:
        df.to_csv(svpath)        

def evaluate_pacs(gpu, modelpath, svpath, backbone, pretrained,projection_dim, channels=3):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Load Model
    if backbone== 'custom':
        if channels == 3:
            cls_net = mnist_net.ConvNet(projection_dim).cuda()
        elif channels == 1:
            cls_net = mnist_net.ConvNet(projection_dim, imdim=channels).cuda()
    elif backbone in ['resnet18','resnet50','wideresnet']:
        if channels == 3:
            encoder = get_resnet(backbone, pretrained= pretrained)
            n_features = encoder.fc.in_features
            output_dim = 7 #pacs
            cls_net = res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda()
        elif channels == 1:
            encoder = get_resnet(backbone, pretrained= pretrained)
            n_features = encoder.fc.in_features
            output_dim = 7 #pacs
            cls_net = res_net.ConvNet(encoder, projection_dim, n_features, output_dim, imdim=channels).cuda()

    saved_weight = torch.load(modelpath) #dict(saved_weight) only has cls_net as key
    cls_net.load_state_dict(saved_weight['cls_net'])
    #cls_net.eval()

    # Test
    str2fun = { 
        'test': data_loader.load_pacs,
        'art': data_loader.load_pacs_acs,
        'cartoon': data_loader.load_pacs_acs,
        'sketch': data_loader.load_pacs_acs
        }   
    columns = ['test','art','cartoon','sketch']
    rst = []
    for data in columns:
        if data == 'test':
            teset = str2fun[data]('test', channels=channels)
        elif data in ['art','cartoon','sketch']:
            teset= str2fun[data](split= data,channels=channels)
        teloader = DataLoader(teset, batch_size=128, num_workers=8,drop_last=True, shuffle=True) #solved error with drop_last/ added shuffle=True(11/25)
        '''
        <Solved Error>
        ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512, 1, 1])
        '''
        teacc = evaluate(cls_net, teloader)
        rst.append(teacc)
        
    
    df = pd.DataFrame([rst], columns=columns)
    print(df)
    if svpath is not None:
        df.to_csv(svpath)
        

if __name__=='__main__':
    main()


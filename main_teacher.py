
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
import copy


from network import mnist_net, res_net, cifar_net, pacs_net
from network.generator import MixStyle
from network.modules import get_resnet, get_generator, freeze, unfreeze, freeze_, unfreeze_, LARS
from main_test import evaluate_digit, evaluate_image, evaluate_pacs, evaluate_officehome
import matplotlib.pyplot as plt
from loss_functions import SupConLoss, MdarLoss
from tools.farmer import *
from tools.miro_utils import *
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
@click.option('--optimizer', type=str, default='adam', help= "adam/sgd")
@click.option('--oracle', type=str, default='True', help= "True/False")
@click.option('--loss_fn', type=str, default='mdar', help= 'Loss Functions (supcon/mdar')
@click.option('--lmda', type=float, default=0.051, help='Lambda for Adversarial BT')
@click.option('--lmda_task', type=float, default=0.0051, help='Lambda for Adversarial BT')
@click.option('--w_oracle', type=float, default=0.1, help='Weight for Professor')
@click.option('--oracle_epoch', type=int, default=10)

def experiment(gpu, data, ntr, translate, autoaug, epochs, nbatch, batchsize, lr, lr_scheduler, svroot, backbone, pretrained, projection_dim, optimizer,oracle,loss_fn,lmda,lmda_task,w_oracle,oracle_epoch):
    settings = locals().copy()
    print(settings)

    # Global Settings
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if not os.path.exists(svroot):
        os.makedirs(svroot)
    writer = SummaryWriter(svroot)
    
    # Load datasets and models
    #MNIST
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
        
    #CIFAR
    elif data in ['cifar10']:
        
        # Load Dataset
        trset = data_loader.load_cifar10(split='train')
        teset = data_loader.load_cifar10(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        
        if backbone == 'custom':
            raise ValueError('PLEASE USE Resnet-18/50 For CIFAR10')
        elif backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as True
            n_features = encoder.fc.in_features #(fc): Linear(in_features=512, out_features=1000, bias=True)
            output_dim= 10
            cls_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() #projection_dim/ n_features
            if optimizer== 'adam':
                cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
            elif optimizer == 'sgd': 
                cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay= 5e-4) #5e-4/1e-1
        elif backbone in ['cifar_net']:
            output_dim= 10
            cls_net= cifar_net.ConvNet(projection_dim=projection_dim, output_dim=output_dim).cuda()
            if optimizer == 'adam':
                cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
            elif optimizer == 'sgd':
                cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
            
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs)
        elif lr_scheduler == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(cls_opt, epochs)
        elif lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(cls_opt, milestones = [60, 120, 160], gamma = 0.2)
    
    #PACS  
    elif data in ['pacs']:
        # Load Dataset
        trset = data_loader.load_pacs(split='train')
        teset = data_loader.load_pacs(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False, drop_last= True)

        if backbone == 'custom':
            raise ValueError('WORK IN PROGRESS: PLEASE USE Resnet-18/50 For PACS')
        elif backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as True
            n_features = encoder.fc.in_features
            output_dim= 7
            cls_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() #projection_dim/ n_features
            if optimizer == 'adam':
                cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
            elif optimizer == 'sgd':
                cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        elif backbone in ['pacs_net']:
            raise ValueError('WORK IN PROGRESS: PLEASE USE Resnet-18/50 For PACS')
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs)
        elif lr_scheduler == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(cls_opt, epochs)

    #OFFICEHOME
    elif data in ['officehome']:
        # Load Dataset
        trset = data_loader.load_officehome(split='train')
        teset = data_loader.load_officehome(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False, drop_last= True)

        if backbone == 'custom':
            raise ValueError('WORK IN PROGRESS: PLEASE USE Resnet-18/50 For Office-Home')
        elif backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as True
            n_features = encoder.fc.in_features
            output_dim= 65
            cls_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() #projection_dim/ n_features
            if optimizer == 'adam':
                cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
            elif optimizer == 'sgd':
                cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs)    
        elif lr_scheduler == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(cls_opt, epochs)
    
    #IMAGENET 
    elif data in ['imagenet']:
        # Load Dataset
        trset = data_loader.load_imagenet(split='train')
        teset = data_loader.load_imagenet(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False, drop_last= True)

        if backbone == 'custom':
            raise ValueError('WORK IN PROGRESS: PLEASE USE Resnet-18/50 For Office-Home')
        elif backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as True
            n_features = encoder.fc.in_features
            output_dim= 1000
            cls_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() #projection_dim/ n_features
            if optimizer == 'adam':
                cls_opt = optim.Adam(cls_net.parameters(), lr=lr)
            elif optimizer == 'sgd':
                cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs)    
        elif lr_scheduler == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(cls_opt, epochs)
    
    
    cls_criterion = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    if loss_fn=='supcon':
        con_criterion = SupConLoss()
    elif loss_fn=='mdar':
        con_criterion = MdarLoss(projection_dim, lmda= lmda,lmda_task=lmda_task)
    cls_net.get_hook()
    
    
    #Create Oracle Model   
    
    #Oracle Net Version.1
    print("--Initializing Teacher Net for Mutual Information Maximization (From Scratch)")
    #Oracle Net version.1
    oracle_net= copy.deepcopy(cls_net)
    prof_net= copy.deepcopy(cls_net)
    freeze("encoder",oracle_net)
    oracle_net.freeze_bn()
    oracle_net= oracle_net.cuda()
    oracle_net.oracle= True
    oracle_net.get_hook()
    
    
    
    #Mean/Variance Encoder Setup
    mean_encoders = None
    var_encoders = None
    # Train Start
    best_acc = 0
    for epoch in range(epochs):
        t1 = time.time()
        
        
        loss_list = []
        cls_net.train()
        
        if epoch == oracle_epoch:
                #set Professor Net
                #Progressive Mutual Information Regularization for Out-of-Domain generalization 
                print("--Initializing Professor Net for Mutual Information Maximization (From Finetuned)")
                #mPATH= os.path.join(svroot,'prof.pt')
                #torch.save(cls_net.state_dict(), mPATH)
                #Professor Net version.1
                #prof_net.load_state_dict(torch.load(mPATH))
                prof_net= copy.deepcopy(cls_net)
                freeze("encoder",prof_net)
                prof_net.freeze_bn()
                prof_net= prof_net.cuda()
                prof_net.oracle= True
                prof_net.get_hook()
        
        
        
        for i, (x, y) in enumerate(trloader):
            x, y = x.cuda(), y.cuda()
            
            # CE
            p = cls_net(x)
            cls_loss = cls_criterion(p, y)
            
            #ORACLE
            if epoch >= oracle_epoch:
                
                
                with torch.no_grad():
                    h_oracle, intermediate_oracle = oracle_net(x, mode= 'encoder_intermediate')
                    h_prof, intermediate_prof = oracle_net(x, mode= 'encoder_intermediate' )
                h_source, intermediate_source = cls_net(x, mode= 'encoder_intermediate')
                
                
                keeper= ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']
                intermediate_oracle= {key: intermediate_oracle[key] for key in keeper}
                intermediate_source= {key: intermediate_source[key] for key in keeper}
                
                
                shapes= [list(intermediate_oracle[i].shape) for i in intermediate_oracle.keys()]
                intermediate_oracle= intermediate_oracle.values()
                intermediate_source= intermediate_source.values()
            
                    
                if not mean_encoders:
                    mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes])
                if not var_encoders:
                    var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes])
                    
                oracle_loss= 0.0

                for f,pre_f, mean_enc, var_enc in zip_strict(intermediate_source,intermediate_oracle,mean_encoders,var_encoders):
                    
                    
                    mean= mean_enc(f) 
                    var= var_enc(f).cuda()
                    
                    vlb= (mean - pre_f).pow(2).div(var) + var.log()
                    oracle_loss += vlb.mean()/2
                    
                prof_tensors= torch.cat([h_prof.unsqueeze(1), h_source.unsqueeze(1)], dim=1)
                prof_loss = con_criterion(prof_tensors, adv=False, standardize= True)    
                
            else:
                oracle_loss= torch.tensor(0)
                prof_loss= torch.tensor(0)
            
            #loss update
            loss= cls_loss + (w_oracle * oracle_loss) + (w_oracle * prof_loss)
            cls_opt.zero_grad()
            loss.backward()
            cls_opt.step()
            
            loss_list.append([cls_loss.item(), oracle_loss.item(), prof_loss.item()])
            
            # Adjust Learning Rate
            if lr_scheduler in ['cosine']:
                scheduler.step()

        cls_loss, oracle_loss, prof_loss = np.mean(loss_list, 0)
        

        # Test and Save Optimal Model
        cls_net.eval()
        if data in ['mnist', 'mnist_t', 'cifar10', 'pacs','officehome']:
            teacc = evaluate(cls_net, teloader)
        '''
        if best_acc < teacc:
            best_acc = teacc
            torch.save({'cls_net':cls_net.state_dict()}, os.path.join(svroot, 'best.pkl'))
        '''
        # Save Log
        t2 = time.time()
        print(f'epoch {epoch}, time {t2-t1:.2f}, cls_loss {cls_loss:.4f} oracle_loss {oracle_loss:.4f} prof_loss {prof_loss:.4f} teacc {teacc:2.2f}')
        writer.add_scalar('scalar/cls_loss', cls_loss, epoch)
        writer.add_scalar('scalar/oracle_loss', oracle_loss, epoch)
        writer.add_scalar('scalar/prof_loss', prof_loss, epoch)
        writer.add_scalar('scalar/teacc', teacc, epoch)
        
    torch.save({'cls_net':cls_net.state_dict()}, os.path.join(svroot, 'best.pkl'))
    writer.close()

def evaluate(net, teloader):
    correct, count = 0, 0
    ps = []
    ys = []
    net.eval() #12/30 midnight - ok
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

class MeanEncoder(nn.Module):
    """Identity function"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, shape, init=0.1, channelwise=True, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                # [B, C, H, W]
                b_shape = (1, shape[1], 1, 1)
            elif len(shape) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            else:
                raise ValueError()

        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, x):
        return F.softplus(self.b) + self.eps

def zip_strict(*iterables):
    """strict version of zip. The length of iterables should be same.
    NOTE yield looks non-reachable, but they are required.
    """
    # For trivial cases, use pure zip.
    if len(iterables) < 2:
        return zip(*iterables)

    # Tail for the first iterable
    first_stopped = False
    def first_tail():
        nonlocal first_stopped
        first_stopped = True
        return
        yield

    # Tail for the zip
    def zip_tail():
        if not first_stopped:
            raise ValueError('zip_equal: first iterable is longer')
        for _ in chain.from_iterable(rest):
            raise ValueError('zip_equal: first iterable is shorter')
            yield

    # Put the pieces together
    iterables = iter(iterables)
    first = chain(next(iterables), first_tail())
    rest = list(map(iter, iterables))
    return chain(zip(first, *rest), zip_tail())

if __name__=='__main__':
    my_seed_everywhere()
    experiment()


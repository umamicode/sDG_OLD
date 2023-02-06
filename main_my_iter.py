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

from loss_functions import SupConLoss, MdarLoss, kl_divergence
from network import mnist_net, res_net, cifar_net, generator
from network.modules import get_resnet, get_generator, freeze, unfreeze, freeze_, unfreeze_, LARS
from tools.miro_utils import *
from tools.farmer import *
import data_loader
from main_base import evaluate

import matplotlib.pyplot as plt


HOME = os.environ['HOME']

@click.command()
@click.option('--gpu', type=str, default='0', help='Choose GPU')
@click.option('--data', type=str, default='mnist', help='Dataset name')
@click.option('--ntr', type=int, default=None, help='Select the first ntr samples of the training set')
@click.option('--gen', type=str, default='cnn', help='cnn/hr')
@click.option('--gen_mode', type=str, default=None, help='Generator Mode')
@click.option('--n_tgt', type=int, default=10, help='Number of Targets')
@click.option('--tgt_epochs', type=int, default=10, help='How many epochs were trained on each target domain')
@click.option('--tgt_epochs_fixg', type=int, default=None, help='When the epoch is greater than this value, G fix is ​​removed')
@click.option('--nbatch', type=int, default=None, help='How many batches are included in each epoch')
@click.option('--batchsize', type=int, default=256)
@click.option('--lr', type=float, default=1e-3, help='Learning Rate: Default 1e-4 in Our Experiment')
@click.option('--lr_scheduler', type=str, default='none', help='Whether to choose a learning rate decay strategy')
@click.option('--svroot', type=str, default='./saved')
@click.option('--ckpt', type=str, default='./saved/best.pkl')
@click.option('--w_cls', type=float, default=1.0, help='cls item weight')
@click.option('--w_info', type=float, default=0.1, help='infomin item weights')
@click.option('--w_cyc', type=float, default=20.0, help='cycleloss item weight')
@click.option('--w_div', type=float, default=2.0, help='Polymorphism loss weight')
@click.option('--w_oracle', type=float, default=1.0, help='Oracle loss Weight')
@click.option('--lmda', type=float, default=0.051, help='Lambda for Adversarial BT')
@click.option('--lmda_task', type=float, default=0.0051, help='Lambda for Adversarial BT')
@click.option('--div_thresh', type=float, default=0.5, help='div_loss threshold')
@click.option('--w_tgt', type=float, default=1.0, help='target domain sample update tasknet intensity control')
@click.option('--interpolation', type=str, default='pixel', help='Interpolate between the source domain and the generated domain to get a new domain, two ways：img/pixel')
@click.option('--loss_fn', type=str, default='supcon', help= 'Loss Functions (supcon/mdar')
@click.option('--backbone', type=str, default= 'custom', help= 'Backbone Model (custom/resnet18,resnet50,wideresnet')
@click.option('--pretrained', type=str, default= 'False', help= 'Pretrained Backbone - ResNet18/50, Custom MNISTnet does not matter')
@click.option('--projection_dim', type=int, default=128, help= "Projection Dimension of the representation vector for Resnet; Default: 128")
@click.option('--oracle', type=str, default='False', help= "Oracle Model for large pretrained models")
@click.option('--optimizer', type=str, default='adam', help= "adam/sgd")



def experiment(gpu, data, ntr, gen, gen_mode, \
        n_tgt, tgt_epochs, tgt_epochs_fixg, nbatch, batchsize, lr, lr_scheduler, svroot, ckpt, \
        w_cls, w_info, w_cyc, w_div, w_oracle,lmda,lmda_task, div_thresh, w_tgt, interpolation, loss_fn, \
        backbone, pretrained, projection_dim, oracle, optimizer):
    settings = locals().copy()
    print(settings)
    print("--Loss Function: {loss_fn}".format(loss_fn= loss_fn))
    print("--Pulling Base Model from: {ckpt}".format(ckpt=ckpt))
    print("--Pushing Trained Model to: {svroot}".format(svroot=svroot))
        
    zdim = 10
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    g1root = os.path.join(svroot, 'g1')
    if not os.path.exists(g1root):
        os.makedirs(g1root)
    writer = SummaryWriter(svroot)
    
    # Load dataset
    imdim = 3 # Default 3 channels
    if data in ['mnist', 'mnist_t', 'mnistvis']:
        if data in [ 'mnist', 'mnistvis']:
            trset = data_loader.load_mnist('train', ntr=ntr)
            teset = data_loader.load_mnist('test')
        elif data == 'mnist_t':
            trset = data_loader.load_mnist_t('train', ntr=ntr)
            teset = data_loader.load_mnist('test')
        imsize = [32, 32]
    elif data in ['cifar10']:
        trset = data_loader.load_cifar10(split='train', autoaug=None) #Autoaug set as None
        teset = data_loader.load_cifar10(split='test')
        imsize = [32, 32]
    elif data in ['pacs']:
        trset = data_loader.load_pacs(split='train')
        teset = data_loader.load_pacs(split='test')
        imsize = [224,224]#[32, 32] 
    elif data in ['officehome']:
        trset = data_loader.load_officehome(split='train')
        teset = data_loader.load_officehome(split='test')
        imsize = [224,224]#[32, 32] 

    print("--Training With {data} data".format(data=data))
    trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))  #WADDUP DUDE 0105 midnight (nbatch*batchsize -> len(trset)) - @ image_randomsampler/ -nope. (bad result)
    teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True) 
    
    # Load model
    g1_list = []
    ### Load Task Model
    ###MNIST
    if data in ['mnist', 'mnist_t']:
        if backbone == 'custom':
            src_net = mnist_net.ConvNet(projection_dim).cuda()
            saved_weight = torch.load(ckpt)
            src_net.load_state_dict(saved_weight['cls_net'])
            src_opt = optim.Adam(src_net.parameters(), lr=lr)
        elif backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as False - We will load our model anyway
            n_features = encoder.fc.in_features
            output_dim = 10 
            src_net= res_net.ConvNet(encoder, projection_dim, n_features,output_dim).cuda() 
            saved_weight = torch.load(ckpt)
            src_net.load_state_dict(saved_weight['cls_net'])
            src_opt = optim.Adam(src_net.parameters(), lr=lr)
    
    ###CIFAR
    elif data in ['cifar10']:
        if backbone == 'custom':
            #NOT RECOMMENDED: CIFAR10 experiment was designed for Resnet Models
            raise ValueError('PLEASE USE Resnet-18/50 For CIFAR10')
        elif backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as False - We will load our model anyway
            n_features = encoder.fc.in_features
            output_dim = 10 
            src_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() #projection_dim/ n_features/output_dim=10
            saved_weight = torch.load(ckpt)
            src_net.load_state_dict(saved_weight['cls_net'])
            if optimizer == 'adam':
                src_opt = optim.Adam(src_net.parameters(), lr=lr)
            elif optimizer == 'sgd':
                src_opt = optim.SGD(src_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-1)
        elif backbone in ['cifar_net']:
            output_dim= 10
            src_net= cifar_net.ConvNet(projection_dim=projection_dim, output_dim=output_dim).cuda()
            saved_weight = torch.load(ckpt)
            src_net.load_state_dict(saved_weight['cls_net'])
            if optimizer == 'adam':
                src_opt = optim.Adam(src_net.parameters(), lr=lr)
            elif optimizer == 'sgd':
                src_opt = optim.SGD(src_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)     
    #PACS
    elif data in ['pacs']:
        if backbone == 'custom':
            #NOT RECOMMENDED: PACS experiment was designed for Resnet Models
            raise ValueError('PLEASE USE Resnet-18/50/AlexNet For PACS')
        elif backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as True
            n_features = encoder.fc.in_features
            output_dim= 7
            src_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() 
            saved_weight = torch.load(ckpt)
            src_net.load_state_dict(saved_weight['cls_net'])
            src_opt = optim.Adam(src_net.parameters(), lr=lr)
    #OfficeHome
    elif data in ['officehome']:
        if backbone in ['resnet18','resnet50','wideresnet']:
            encoder = get_resnet(backbone, pretrained) # Pretrained Backbone default as True
            n_features = encoder.fc.in_features
            output_dim= 65
            src_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() 
            saved_weight = torch.load(ckpt)
            src_net.load_state_dict(saved_weight['cls_net'])
            src_opt = optim.Adam(src_net.parameters(), lr=lr)

    cls_criterion = nn.CrossEntropyLoss()
    
    if loss_fn=='supcon':
        con_criterion = SupConLoss()
    elif loss_fn=='mdar':
        con_criterion = MdarLoss(projection_dim, lmda= lmda,lmda_task=lmda_task)
    
    #kl_loss = nn.KLDivLoss(reduction="batchmean")
    ##########################################    
    #Create Oracle Model    
    if (oracle == 'True'):

        #Oracle Net version.1
        print("--Initializing Teacher Net for Mutual Information Maximization (From Finetuned)")
        oracle_net= copy.deepcopy(src_net)
        freeze("encoder",oracle_net)
        oracle_net= oracle_net.cuda()
        oracle_net.oracle= True
        
        #Oracle Net Version.2
        '''
        print("--Initializing Teacher Net for Mutual Information Maximization (From Scratch)")
        encoder = get_resnet(backbone, pretrained='True') # Pretrained Backbone default as False - We will load our model anyway
        n_features = encoder.fc.in_features
        output_dim = output_dim #10 
        oracle_net= res_net.ConvNet(encoder, projection_dim, n_features, output_dim).cuda() 
        freeze("encoder",oracle_net)
        oracle_net.oracle= True
        '''
        #Get Hooks (Currently only in resnet)
        oracle_net.get_hook()
        src_net.get_hook()
    
    #Mean, Variance Encoder
    mean_encoders = None
    var_encoders = None
    ##########################################
    # Train
    global_best_acc = 0
    for i_tgt in range(n_tgt):
        print(f'target domain {i_tgt}/{n_tgt}')

        #ith target generator train sequence
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(src_opt, tgt_epochs*len(trloader)) #(https://discuss.pytorch.org/t/cosineannealinglr-step-size-t-max/104687/)
        elif lr_scheduler == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(src_opt, tgt_epochs)
        
        
        g1_net, g2_net, g1_opt, g2_opt = get_generator(gen, imdim=imdim, imsize= imsize, lr= lr) #get_generator(gen)
        best_acc = 0
        for epoch in range(tgt_epochs):
            t1 = time.time()
            
            # if flag_fixG = False, locking G / flag_fixG = True, renew G
            flag_fixG = False
            if (tgt_epochs_fixg is not None) and (epoch >= tgt_epochs_fixg):
                flag_fixG = True
            loss_list = []
            time_list = []
            
            
            src_net.train() 
            #src_net.eval()
            for i, (x, y) in enumerate(trloader):  

                x, y = x.cuda(), y.cuda()
        
                # Data Augmentation
                if len(g1_list)>0: # if generator exists (not zero in g1_list)
                    idx = np.random.randint(0, len(g1_list))
                    if gen in ['hr', 'cnn']:
                        with torch.no_grad():
                            x2_src = g1_list[idx](x, rand=True)  #generated image
                        # domain interpolation
                        if interpolation == 'img':
                            rand = torch.rand(len(x), 1, 1, 1).cuda()
                            x3_mix = rand*x + (1-rand)*x2_src  #add random noise
                    elif gen == 'stn':
                        with torch.no_grad():
                            x2_src, H = g1_list[idx](x, rand=True, return_H=True)  #stn generated image
                        # domain interpolation
                        if interpolation == 'H':
                            rand = torch.rand(len(x), 1, 1).cuda()
                            std_H = torch.tensor([[1, 0, 0], [0, 1, 0]]).float().cuda()
                            H = rand*std_H + (1-rand)*H
                            grid = F.affine_grid(H, x.size())
                            x3_mix = F.grid_sample(x, grid)

                # Synthesize new data
                if gen in ['cnn', 'hr']:
                    x_tgt = g1_net(x, rand=True)
                    x2_tgt = g1_net(x, rand=True)
                    
                elif gen == 'stn':
                    x_tgt, H_tgt = g1_net(x, rand=True, return_H=True)
                    x2_tgt, H2_tgt = g1_net(x, rand=True, return_H=True)


                # forward
                p1_src, z1_src = src_net(x, mode='train') #z1- torch.Size([128, 128])
                
                
                # oracle forward
                if (oracle == 'True'):                                       
                    #For Domain Alignment
                    #h_oracle= oracle_net(x,mode='encoder')#oracle
                    #h_source= src_net(x,mode='encoder')#oracle    
                    
                    #after dinner
                    h_oracle, intermediate_oracle = oracle_net(x, mode= 'encoder_intermediate')
                    h_source, intermediate_source = src_net(x, mode= 'encoder_intermediate')
                    
                    keeper= ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']
                    intermediate_oracle= {key: intermediate_oracle[key] for key in keeper} #F.normalize run47
                    intermediate_source= {key: intermediate_source[key] for key in keeper} #F.normalize run47
            
            
                    shapes= [list(intermediate_oracle[i].shape) for i in intermediate_oracle.keys()]
                    intermediate_oracle= intermediate_oracle.values()
                    intermediate_source= intermediate_source.values()
                    
                    #Generate Mean/Variance Encoders
                    if not mean_encoders:
                        mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes])
                    if not var_encoders:
                        var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes])
                    
                    
                
                if len(g1_list)>0: # if generator exists
                    p2_src, z2_src = src_net(x2_src, mode='train') #z2- torch.Size([128, 128])
                    p3_mix, z3_mix = src_net(x3_mix, mode='train') #z3- torch.Size([128, 128])
                    
                    zsrc = torch.cat([z1_src.unsqueeze(1), z2_src.unsqueeze(1), z3_mix.unsqueeze(1)], dim=1) #OG
                    
                    src_cls_loss = cls_criterion(p1_src, y) + cls_criterion(p2_src, y) + cls_criterion(p3_mix, y)  #GreatCloneDetach GCD

                else:
                    zsrc = z1_src.unsqueeze(1)                       
                    src_cls_loss = cls_criterion(p1_src, y) #GCD

                p_tgt, z_tgt = src_net(x_tgt, mode='train')
                tgt_cls_loss = cls_criterion(p_tgt, y) #[TODO] GCD
                                
                
                # SRC-NET UPDATE
                zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1) #OG
                con_loss = con_criterion(zall, adv=False)
                
                #oracle
                if (oracle == 'True'):
                    #oracle_tensors are not normalized (dim=1).
                    #oracle_tensors= torch.cat([h_oracle.unsqueeze(1), h_source.unsqueeze(1)], dim=1)
                    #oracle_loss = con_criterion(oracle_tensors, adv=False, standardize= True) #standardize true showed better results
                    
                    oracle_loss= 0.0
                    #og oracle
                    for f, pre_f, mean_enc, var_enc in zip_strict(intermediate_source,intermediate_oracle,mean_encoders,var_encoders):
                        mean= mean_enc(f) 
                        var= var_enc(f).cuda() #idk why but not in gpu
                        vlb= (mean - pre_f).pow(2).div(var) + var.log()
                        oracle_loss += vlb.mean()/ 2.
                    
                
                #Source Task Model Loss
                loss = src_cls_loss + w_tgt*tgt_cls_loss + w_tgt*con_loss  #og
                
                #Oracle Loss 
                if (oracle == 'True'):
                    loss += (w_oracle* oracle_loss)
                elif (oracle == 'False'):
                    oracle_loss= torch.tensor(0)
                
                src_opt.zero_grad()
                if flag_fixG:
                    loss.backward() #og
                else:
                    loss.backward(retain_graph=True) #og
                
                
                # G1-NET UPDATE
                if flag_fixG:
                    # fix G，training only tasknet
                    con_loss_adv = torch.tensor(0)
                    div_loss = torch.tensor(0)
                    cyc_loss = torch.tensor(0)
                    #Update Source Net (RUN1)
                    src_opt.step() 
                
                else:
                    idx = np.random.randint(0, zsrc.size(1))
                    zall = torch.cat([z_tgt.unsqueeze(1), zsrc[:,idx:idx+1].detach()], dim=1)
                    con_loss_adv = con_criterion(zall, adv=True) #[TODO]GCD 
                    
                    if gen in ['cnn', 'hr']:
                        div_loss = (x_tgt-x2_tgt).abs().mean([1,2,3]).clamp(max=div_thresh).mean() # Constraint Generator Divergence
                        x_tgt_rec = g2_net(x_tgt)
                        cyc_loss = F.mse_loss(x_tgt_rec, x) 
                    elif gen == 'stn':
                        div_loss = (H_tgt-H2_tgt).abs().mean([1,2]).clamp(max=div_thresh).mean()
                        cyc_loss = torch.tensor(0).cuda()
                    
                    if loss_fn == 'mdar':
                        loss = w_cls*tgt_cls_loss - w_div*div_loss + w_cyc*cyc_loss - w_info*con_loss_adv #og
                    elif loss_fn == 'supcon':
                        loss = w_cls*tgt_cls_loss - w_div*div_loss + w_cyc*cyc_loss + w_info*con_loss_adv
                    '''
                    - Error: RuntimeError: one of the variables needed for gradient computation
                    has been modified by an inplace operation: [torch.cuda.FloatTensor [128, 128]], 
                    which is output 0 of AsStridedBackward0, is at version 3; expected version 2 instead.
                    - Problem: both tgt_cls_loss and con_loss_adv is causing the problem
                    - Solution: step() at once. 
                    '''
                    
                    g1_opt.zero_grad()
                    if g2_opt is not None:
                        g2_opt.zero_grad()
                    loss.backward()
                    
                    #Update Source Net (RUN1)
                    src_opt.step() 
                    g1_opt.step()
                    if g2_opt is not None:
                        g2_opt.step()
                    
                #src_opt.step() 
                 
                # update learning rate
                if lr_scheduler in ['cosine']:
                    scheduler.step()
               
                loss_list.append([src_cls_loss.item(), tgt_cls_loss.item(), con_loss.item(), con_loss_adv.item(), div_loss.item(), cyc_loss.item(), oracle_loss.item()])
            src_cls_loss, tgt_cls_loss, con_loss, con_loss_adv, div_loss, cyc_loss, oracle_loss = np.mean(loss_list, 0)
            
            # Test
            src_net.eval()
            
            # unified teacc
            if data in ['mnist', 'mnist_t', 'mnistvis','cifar10','pacs','officehome']:
                teacc = evaluate(src_net, teloader) 
            
            #Save Best Model
            if best_acc < teacc:
                best_acc = teacc
                torch.save({'cls_net':src_net.state_dict()}, os.path.join(svroot, f'{i_tgt}-best.pkl'))
            #if global_best_acc < teacc:
            #    global_best_acc = teacc
            #    torch.save({'cls_net':src_net.state_dict()}, os.path.join(svroot, f'best.pkl'))

            t2 = time.time()

            # Save Log for Tensorboard
            print(f'epoch {epoch}, time {t2-t1:.2f}, src_cls {src_cls_loss:.4f} tgt_cls {tgt_cls_loss:.4f} con {con_loss:.4f} con_adv {con_loss_adv:.4f} div {div_loss:.4f} cyc {cyc_loss:.4f} oracle {oracle_loss:.4f} /// teacc {teacc:2.2f}')
            writer.add_scalar('scalar/src_cls_loss', src_cls_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/tgt_cls_loss', tgt_cls_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/con_loss', con_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/con_loss_adv', con_loss_adv, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/div_loss', div_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/cyc_loss', cyc_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/oracle_loss', oracle_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/teacc', teacc, i_tgt*tgt_epochs+epoch)
            
            
            g1_all = g1_list + [g1_net]
            x = x[0:10]
            l1 = make_grid(x, 1, 2, pad_value=128) 
            l_list = [l1]
            with torch.no_grad():
                for i in range(len(g1_all)):
                    #rand = torch.randn(len(x), zdim).cuda()
                    x_ = g1_all[i](x, rand=True)
                    l_list.append(make_grid(x_, 1, 2, pad_value=128))
                if g2_net is not None:
                    x_x = g2_net(x_)
                    l_list.append(make_grid(x_x, 1, 2, pad_value=128))
                rst = make_grid(torch.stack(l_list), len(l_list), pad_value=128)
                writer.add_image('im-gen', rst, i_tgt*tgt_epochs+epoch)

                x_copy = x[0:1].repeat(16, 1, 1, 1)
                x_copy_ = g1_net(x_copy, rand=True)
                rst = make_grid(x_copy_, 4, 2, pad_value=128)
                writer.add_image('im-div', rst, i_tgt*tgt_epochs+epoch)
            if len(g1_list)>0:
                l1 = make_grid(x[0:6], 6, 2, pad_value=128)
                l2 = make_grid(x2_src[0:6], 6, 2, pad_value=128)
                l3 = make_grid(x3_mix[0:6], 6, 2, pad_value=128)
                rst = make_grid(torch.stack([l1, l3, l2]), 1, pad_value=128)
                writer.add_image('im-mix', rst, i_tgt*tgt_epochs+epoch)

        # Save trained G1(Generator)
        torch.save({'g1':g1_net.state_dict()}, os.path.join(g1root, f'{i_tgt}.pkl'))
        
        g1_list.append(g1_net)
        
        
        # Test the generalization effect of the i_tgt model - (run1)

        from main_test import evaluate_digit, evaluate_image, evaluate_pacs, evaluate_officehome
        
        if data == 'mnist':
            pklpath = f'{svroot}/{i_tgt}-best.pkl'
            evaluate_digit(gpu, pklpath, pklpath+'.test', backbone= backbone, pretrained= pretrained, projection_dim= projection_dim) #Pretrained set as False, it will load our model instead.
        elif data == 'cifar10':
            pklpath = f'{svroot}/{i_tgt}-best.pkl'
            evaluate_image(gpu, pklpath, pklpath+'.test', backbone= backbone, pretrained= pretrained, projection_dim= projection_dim, c_level= 5)
        elif data == 'pacs':
            pklpath = f'{svroot}/{i_tgt}-best.pkl'
            evaluate_pacs(gpu, pklpath, pklpath+'.test', backbone= backbone, pretrained= pretrained, projection_dim= projection_dim)
        elif data == 'officehome':
            pklpath = f'{svroot}/{i_tgt}-best.pkl'
            evaluate_officehome(gpu, pklpath, pklpath+'.test', backbone= backbone, pretrained= pretrained, projection_dim= projection_dim)
    writer.close()

if __name__=='__main__':
    my_seed_everywhere()
    experiment()


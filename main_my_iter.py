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

from con_losses import SupConLoss, ReLICLoss
from network import mnist_net, generator
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
@click.option('--lr', type=float, default=1e-3)
@click.option('--lr_scheduler', type=str, default='none', help='Whether to choose a learning rate decay strategy')
@click.option('--svroot', type=str, default='./saved')
@click.option('--ckpt', type=str, default='./saved/best.pkl')
@click.option('--w_cls', type=float, default=1.0, help='cls item weight')
@click.option('--w_info', type=float, default=1.0, help='infomin item weights')
@click.option('--w_cyc', type=float, default=10.0, help='cycleloss item weight')
@click.option('--w_div', type=float, default=1.0, help='Polymorphism loss weight')
@click.option('--div_thresh', type=float, default=0.1, help='div_loss threshold')
@click.option('--w_tgt', type=float, default=1.0, help='target domain sample update tasknet intensity control')
@click.option('--interpolation', type=str, default='pixel', help='Interpolate between the source domain and the generated domain to get a new domain, two ways：img/pixel')
@click.option('--relic/--no-relic', default=False)

def experiment(gpu, data, ntr, gen, gen_mode, \
        n_tgt, tgt_epochs, tgt_epochs_fixg, nbatch, batchsize, lr, lr_scheduler, svroot, ckpt, \
        w_cls, w_info, w_cyc, w_div, div_thresh, w_tgt, interpolation, relic):
    settings = locals().copy()
    print(settings)
    print("--Relic: {relic}\n".format(relic= relic))
    
    # Global Settings
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
    trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))
    teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
    
    # load model
    def get_generator(name):  #[TODO]maybe not gen but name?
        if name=='cnn':
            g1_net = generator.cnnGenerator(imdim=imdim, imsize=imsize).cuda()
            g2_net = generator.cnnGenerator(imdim=imdim, imsize=imsize).cuda()
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr)
            g2_opt = optim.Adam(g2_net.parameters(), lr=lr)
        elif gen=='hr':
            1/0
            g1_net = hrnet.HRGenerator(zdim=zdim).cuda()
            g2_net = hrnet.HRGenerator(zdim=zdim).cuda()
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr)
            g2_opt = optim.Adam(g2_net.parameters(), lr=lr)
        elif gen=='stn':
            g1_net = generator.stnGenerator(zdim=zdim, mode=gen_mode).cuda()
            g2_net = None
            g1_opt = optim.Adam(g1_net.parameters(), lr=lr/2)
            g2_opt = None
        return g1_net, g2_net, g1_opt, g2_opt

    g1_list = []
    ### Load Model ([TODO]- Add Mutual Information Regularization Method for Pretrained Models)
    if data in ['mnist', 'mnist_t']:
        src_net = mnist_net.ConvNet().cuda()
        saved_weight = torch.load(ckpt)
        src_net.load_state_dict(saved_weight['cls_net'])
        src_opt = optim.Adam(src_net.parameters(), lr=lr)

    elif data == 'mnistvis':
        src_net = mnist_net.ConvNetVis().cuda()
        saved_weight = torch.load(ckpt)
        src_net.load_state_dict(saved_weight['cls_net'])
        src_opt = optim.Adam(src_net.parameters(), lr=lr)

    cls_criterion = nn.CrossEntropyLoss()
    ##########################################
    #[TODO]- Add ReLIC LOSS (221112)
    if relic==False:
        con_criterion = SupConLoss()
    elif relic==True:
        con_criterion = ReLICLoss()
    ##########################################    
    
    # Train
    global_best_acc = 0
    for i_tgt in range(n_tgt):
        print(f'target domain {i_tgt}/{n_tgt}')

        ####################### Learning ith target generator
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(src_opt, tgt_epochs*len(trloader))
        g1_net, g2_net, g1_opt, g2_opt = get_generator(gen)
        best_acc = 0
        for epoch in range(tgt_epochs):
            t1 = time.time()
            
            # if flag_fixG = False, locking G
            #      flag_fixG = True, renew G
            flag_fixG = False
            if (tgt_epochs_fixg is not None) and (epoch >= tgt_epochs_fixg):
                flag_fixG = True
            loss_list = []
            time_list = []
            #src_net.train()
            src_net.eval()
            for i, (x, y) in enumerate(trloader):  
                x, y = x.cuda(), y.cuda()

                # Data Augmentation
                if len(g1_list)>0: # if generator exists (not zero in g1_list)
                    idx = np.random.randint(0, len(g1_list))
                    #rand = torch.randn(len(x), zdim).cuda()
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
                #rand = torch.randn(len(x), zdim).cuda()
                #rand2 = torch.randn(len(x), zdim).cuda()
                if gen in ['cnn', 'hr']:
                    x_tgt = g1_net(x, rand=True)
                    x2_tgt = g1_net(x, rand=True)
                    
                elif gen == 'stn':
                    x_tgt, H_tgt = g1_net(x, rand=True, return_H=True)
                    x2_tgt, H2_tgt = g1_net(x, rand=True, return_H=True)
                
                

                # forward
                p1_src, z1_src = src_net(x, mode='train') #z1- torch.Size([128, 128])
                if len(g1_list)>0: # if generator exists
                    p2_src, z2_src = src_net(x2_src, mode='train') #z2- torch.Size([128, 128])
                    p3_mix, z3_mix = src_net(x3_mix, mode='train') #z3- torch.Size([128, 128])
                    zsrc = torch.cat([z1_src.unsqueeze(1), z2_src.unsqueeze(1), z3_mix.unsqueeze(1)], dim=1) #zsrc- torch.Size([128, 3, 128])
                    
                    #src_cls_loss = cls_criterion(p1_src, y) + cls_criterion(p2_src, y) + cls_criterion(p3_mix, y)
                    src_cls_loss = cls_criterion(p1_src, y) + cls_criterion(p2_src, y) + cls_criterion(p3_mix, y)  #{TODO} GreatCloneDetach GCD

                else:
                    zsrc = z1_src.unsqueeze(1)   #[TODO] Tried This and worked
                    #src_cls_loss = cls_criterion(p1_src, y) #[TODO] GCD                    
                    src_cls_loss = cls_criterion(p1_src, y) #[TODO] GCD

                p_tgt, z_tgt = src_net(x_tgt, mode='train')
                #tgt_cls_loss = cls_criterion(p_tgt, y) #[TODO] GCD
                tgt_cls_loss = cls_criterion(p_tgt.clone().detach(), y) #[TODO] GCD
                
                ######TODO[Change ORDER G & F]
                ##'''
                ##[TODO]- G(G1/2_opt): CHANGE ORDER WITH F###
                # update g1_net
                if flag_fixG:
                    # fix G，training only tasknet
                    con_loss_adv = torch.tensor(0)
                    div_loss = torch.tensor(0)
                    cyc_loss = torch.tensor(0)
                else:
                    idx = np.random.randint(0, zsrc.size(1))
                    zall = torch.cat([z_tgt.unsqueeze(1), zsrc[:,idx:idx+1].detach()], dim=1)
                    #con_loss_adv = con_criterion(zall, adv=True) #[TODO ]GCD
                    
                    ##########################################
                    #[TODO]- Add ReLIC LOSS for Generator(G1) (221112)
                    # Takes {zall = torch.cat([z_tgt.unsqueeze(1), zsrc[:,idx:idx+1].detach()], dim=1)} as input.
                    if relic ==False:
                        con_loss_adv = con_criterion(zall.clone().detach(), adv=True) #[TODO ]GCD
                    elif relic == True:
                        con_loss_adv = con_criterion(zall.clone().detach(), adv=True) #[TODO ]GCD
                    ##########################################
                    
                    if gen in ['cnn', 'hr']:
                        div_loss = (x_tgt-x2_tgt).abs().mean([1,2,3]).clamp(max=div_thresh).mean() # Constraint Generator Divergence
                        x_tgt_rec = g2_net(x_tgt)
                        cyc_loss = F.mse_loss(x_tgt_rec, x)
                    elif gen == 'stn':
                        div_loss = (H_tgt-H2_tgt).abs().mean([1,2]).clamp(max=div_thresh).mean()
                        cyc_loss = torch.tensor(0).cuda()
                    loss = w_cls*tgt_cls_loss - w_div*div_loss + w_cyc*cyc_loss + w_info*con_loss_adv
                    g1_opt.zero_grad()
                    if g2_opt is not None:
                        g2_opt.zero_grad()
                    loss.backward()
                    g1_opt.step()
                    if g2_opt is not None:
                        g2_opt.step()

                ###[TODO]- F(src_net): CHANGE ORDER WITH G###
                # update src_net
                zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1) #torch.Size([128, num_generated_domains, 128])
                
                #con_loss = con_criterion(zall, adv=False) #[TODO] GCD
                '''
                The original version caused error:
                https://discuss.pytorch.org/t/83241
                Fixed by clone&detaching tensors
                '''
                ##########################################
                #[TODO]- Add ReLIC LOSS for Task Model(src_net) (221112)
                # Takes zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1) as input.
                if relic==False:
                    con_loss = con_criterion(zall.clone().detach(), adv=False) #[TODO] GCD
                elif relic== True:
                    con_loss = con_criterion(zall.clone().detach(), adv=False) #[TODO] GCD
                ##########################################

                loss = src_cls_loss + w_tgt*con_loss + w_tgt*tgt_cls_loss # w_tgt default 1.0
                src_opt.zero_grad()
                if flag_fixG:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward(retain_graph=True)
                src_opt.step()     

                #Ends- Change Order (G/F)
                ##'''       
                '''
                ###[TODO]- F(src_net): CHANGE ORDER WITH G###
                # update src_net
                zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1)
                con_loss = con_criterion(zall, adv=False)
                loss = src_cls_loss + w_tgt*con_loss + w_tgt*tgt_cls_loss # w_tgt default 1.0
                src_opt.zero_grad()
                if flag_fixG:
                    loss.backward()
                else:
                    loss.backward()
                src_opt.step()
                
                ###[TODO]- G(G1/2_opt): CHANGE ORDER WITH F###
                # update g1_net
                if flag_fixG:
                    # fix G，training only tasknet
                    con_loss_adv = torch.tensor(0)
                    div_loss = torch.tensor(0)
                    cyc_loss = torch.tensor(0)
                else:
                    idx = np.random.randint(0, zsrc.size(1))
                    zall = torch.cat([z_tgt.unsqueeze(1), zsrc[:,idx:idx+1].detach()], dim=1)
                    con_loss_adv = con_criterion(zall, adv=True)
                    if gen in ['cnn', 'hr']:
                        div_loss = (x_tgt-x2_tgt).abs().mean([1,2,3]).clamp(max=div_thresh).mean() # Constraint Generator Divergence
                        x_tgt_rec = g2_net(x_tgt)
                        cyc_loss = F.mse_loss(x_tgt_rec, x)
                    elif gen == 'stn':
                        div_loss = (H_tgt-H2_tgt).abs().mean([1,2]).clamp(max=div_thresh).mean()
                        cyc_loss = torch.tensor(0).cuda()
                    loss = w_cls*tgt_cls_loss - w_div*div_loss + w_cyc*cyc_loss + w_info*con_loss_adv
                    g1_opt.zero_grad()
                    if g2_opt is not None:
                        g2_opt.zero_grad()
                    loss.backward()
                    g1_opt.step()
                    if g2_opt is not None:
                        g2_opt.step()
            '''
                # update learning rate
                if lr_scheduler in ['cosine']:
                    scheduler.step()
               
                loss_list.append([src_cls_loss.item(), tgt_cls_loss.item(), con_loss.item(), con_loss_adv.item(), div_loss.item(), cyc_loss.item()])
            src_cls_loss, tgt_cls_loss, con_loss, con_loss_adv, div_loss, cyc_loss = np.mean(loss_list, 0)
            
            # Test
            src_net.eval()
            # mnist、cifar test process synthia is different
            if data in ['mnist', 'mnist_t', 'mnistvis']:
                teacc = evaluate(src_net, teloader)
            if best_acc < teacc:
                best_acc = teacc
                torch.save({'cls_net':src_net.state_dict()}, os.path.join(svroot, f'{i_tgt}-best.pkl'))
            #if global_best_acc < teacc:
            #    global_best_acc = teacc
            #    torch.save({'cls_net':src_net.state_dict()}, os.path.join(svroot, f'best.pkl'))

            t2 = time.time()

            # Save Log for Tensorboard
            print(f'epoch {epoch}, time {t2-t1:.2f}, src_cls {src_cls_loss:.4f} tgt_cls {tgt_cls_loss:.4f} con {con_loss:.4f} con_adv {con_loss_adv:.4f} div {div_loss:.4f} cyc {cyc_loss:.4f} /// teacc {teacc:2.2f}')
            writer.add_scalar('scalar/src_cls_loss', src_cls_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/tgt_cls_loss', tgt_cls_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/con_loss', con_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/con_loss_adv', con_loss_adv, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/div_loss', div_loss, i_tgt*tgt_epochs+epoch)
            writer.add_scalar('scalar/cyc_loss', cyc_loss, i_tgt*tgt_epochs+epoch)
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
                #rand = torch.randn(16, zdim).cuda()
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

        # Test the generalization effect of the i_tgt model
        from main_test_digit import evaluate_digit
        if data == 'mnist':
            pklpath = f'{svroot}/{i_tgt}-best.pkl'
            evaluate_digit(gpu, pklpath, pklpath+'.test')

    writer.close()

if __name__=='__main__':
    experiment()


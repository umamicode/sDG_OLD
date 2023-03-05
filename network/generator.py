#Reference: https://github.com/lileicv/PDEN/blob/main/network/generator.py by @lileicv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.modules.batchinstance_norm import BatchInstanceNorm2d
import random

class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        #self.norm = nn.InstanceNorm2d(num_features, affine=False)  
        self.norm = BatchInstanceNorm2d(num_features, affine = True)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s): 
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta
        

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)
    

class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix



class cnnGenerator(nn.Module): #added noise after breakup
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[192, 320]):
        ''' w_ln local noise weight
        '''
        super().__init__()
        stride = (kernelsize-1)//2
        self.zdim = zdim = 10
        self.imdim = imdim
        self.imsize = imsize

        self.conv1 = nn.Conv2d(imdim, n, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(n, 2*n, kernelsize, 1, stride)
        self.adain2 = AdaIN2d(zdim, 2*n)
        self.conv3 = nn.Conv2d(2*n, 4*n, kernelsize, 1, stride)
        self.conv4 = nn.Conv2d(4*n, imdim, kernelsize, 1, stride)
        self.mixstyle= MixStyle(mix='random') #crossdomain
        
        #STN
        self.mapz = nn.Linear(zdim, imsize[0]*imsize[1])
        if imsize == [32,32]:
            self.loc = nn.Sequential(
                    nn.Conv2d( 3,  16, 5), nn.MaxPool2d(2), nn.ReLU(),
                    #nn.Conv2d( 4,  16, 5), nn.MaxPool2d(2), nn.ReLU(), #MIDNIGHT
                    nn.Conv2d( 16, 32, 5), nn.MaxPool2d(2), nn.ReLU(),)
            self.fc_loc = nn.Sequential(
                    nn.Linear(32*5*5, 32), nn.ReLU(),
                    nn.Linear(32, 6))
        elif imsize == [224,224]:
            self.loc = nn.Sequential(
                    nn.Conv2d(3,16,5,2), nn.MaxPool2d(2), nn.ReLU(),
                    #nn.Conv2d(4,16,5,2), nn.MaxPool2d(2), nn.ReLU(), #MIDNIGHT
                    nn.Conv2d(16,32,5,2), nn.MaxPool2d(2), nn.ReLU(),
                    nn.Conv2d(32,32,5,2),nn.ReLU()
            )
            self.fc_loc = nn.Sequential(
                    nn.Linear(32*5*5, 32), nn.ReLU(),
                    nn.Linear(32, 6))
        # weight initialization
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0]))

    def forward(self, x, rand=False, return_H= False): 
        ''' x '''
        '''       
        #STN
        z = torch.randn(len(x), self.zdim).cuda()
        z = self.mapz(z).view(len(x), 1, x.size(2), x.size(3))
        loc = self.loc(torch.cat([x, z], dim=1)) # [N, -1]
        loc = loc.view(len(loc), -1)
        H = self.fc_loc(loc)
        H = H.view(len(H), 2, 3)
        '''
        #H[:,0,0] = 1 
        #H[:,0,1] = 0 
        #H[:,1,0] = 0 
        #H[:,1,1] = 1 
        '''
        grid = F.affine_grid(H, x.size())
        x = F.grid_sample(x, grid)
        '''
        
        loc = self.loc(x)
        loc = loc.view(len(loc), -1)
        H = self.fc_loc(loc)
        H = H.view(len(H), 2, 3)
        
        H[:,0,0] = 1 
        H[:,0,1] = 0 
        H[:,1,0] = 0 
        H[:,1,1] = 1 
        
        grid = F.affine_grid(H, x.size())
        x = F.grid_sample(x, grid)
        
        
        #MIXSTYLE + Style-Transfer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x= self.mixstyle(x)
        
        if rand:
            z = torch.randn(len(x), self.zdim).cuda() #run1
            #z = torch.randn(1, self.zdim).cuda() #run0
            #z= z.repeat(len(x), 1) #run0
            x= self.mixstyle(x)
            x = self.adain2(x, z)
        x = F.relu(self.conv3(x))
        x= self.mixstyle(x)
        x = torch.sigmoid(self.conv4(x))
                
        
        if return_H:
            return x, H
        else:
            return x



class stnGenerator(nn.Module):
    ''' Affine transformation '''
    def __init__(self, zdim=10, imsize=[32,32], mode=None):
        super().__init__()
        self.mode = mode
        self.zdim = zdim
        
        self.mapz = nn.Linear(zdim, imsize[0]*imsize[1])
        if imsize == [32,32]:
            self.loc = nn.Sequential(
                    nn.Conv2d( 4,  16, 5), nn.MaxPool2d(2), nn.ReLU(),
                    nn.Conv2d( 16, 32, 5), nn.MaxPool2d(2), nn.ReLU(),)
            self.fc_loc = nn.Sequential(
                    nn.Linear(32*5*5, 32), nn.ReLU(),
                    nn.Linear(32, 6))
        # weight initialization
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0]))
    def forward(self, x, rand, return_H=False):
        if rand:
            z = torch.randn(len(x), self.zdim).cuda()
        z = self.mapz(z).view(len(x), 1, x.size(2), x.size(3))
        loc = self.loc(torch.cat([x, z], dim=1)) # [N, -1]
        loc = loc.view(len(loc), -1)
        H = self.fc_loc(loc)
        H = H.view(len(H), 2, 3)
        if self.mode == 'translate':
            H[:,0,0] = 1 
            H[:,0,1] = 0 
            H[:,1,0] = 0 
            H[:,1,1] = 1 
        grid = F.affine_grid(H, x.size())
        x = F.grid_sample(x, grid)
        if return_H:
            return x, H
        else:
            return x

if __name__=='__main__':
    x = torch.ones(4, 3, 32, 32)
    z = torch.ones(4, 10)
    
    g = stnGenerator(10, [32, 32])
    y = g(x, z)



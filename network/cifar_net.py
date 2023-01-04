import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import sys
import numpy as np

from network.modules.resnet_hacks import modify_resnet_model
from network.modules.identity import Identity
from collections import OrderedDict

#[TODO]-added for check
torch.autograd.set_detect_anomaly(True)

def freeze_(model):
    """Freeze model
    Note that this function does not control BN
    """
    for p in model.parameters():
        p.requires_grad_(False)


#WIDE-RESNET by @meliketoy (reference: https://github.com/meliketoy/wide-resnet.pytorch)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class ConvNet(nn.Module):
    ''' The network structure is consistent with the SimCLR method
     '''
    def __init__(self, projection_dim, output_dim, imdim=3, oracle= False):
        super(ConvNet, self).__init__()
        
        #added 
        self.encoder= Wide_ResNet(depth=16, widen_factor=4, dropout_rate=0.3, num_classes= output_dim) #16,4
        self.projection_dim= projection_dim
        self.n_features= self.encoder.linear.in_features
        self.output_dim= output_dim
        #added for miro
        self.oracle= oracle
        self.selected_out = OrderedDict()
        self.fhooks=[]
        
        # Replace the fc layer with an Identity function
        self.encoder.linear = Identity()
        
        self.cls_head_src = nn.Linear(self.n_features, self.output_dim)
        #self.cls_head_src= nn.Sequential(nn.Linear(self.n_features, self.n_features), nn.BatchNorm1d(self.n_features),nn.ReLU(),nn.Linear(self.n_features, self.output_dim))
        self.cls_head_tgt = nn.Linear(self.n_features, self.output_dim)
        #[TODO]- MLP for Contrastive Learning -Following model design of BarlowTwins Paper
        self.pro_head = nn.Sequential(
            nn.Linear(self.n_features, self.n_features*2, bias=False),  #self.n_features -> self.projection_dim
            nn.BatchNorm1d(self.n_features*2),
            nn.ReLU(),
            nn.Linear(self.n_features*2, self.n_features*2, bias=False),  #self.n_features -> self.projection_dim
            nn.BatchNorm1d(self.n_features*2),
            nn.ReLU(),
            nn.Linear(self.n_features*2, self.projection_dim, bias=False), #self.n_features,self.projection_dim -> self.projection_dim,self.projection_dim
        )
        
    
    def get_hook(self):   
        for i,l in enumerate(list(self.encoder._modules.keys())):
            self.fhooks.append(getattr(self.encoder,l).register_forward_hook(self.forward_hook(l)))
        
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
    
    
        
    def forward(self, x, mode='test'):
        in_size = x.size(0)
        out4= self.encoder(x)
        #F.log_softmax(self.cls_head_src(out4), dim=-1)
        if mode == 'test':
            #p = self.cls_head_src(out4)
            p= self.cls_head_src(out4)
            return p
        elif mode == 'train':
            #p = self.cls_head_src(out4)
            p= self.cls_head_src(out4)
            z = self.pro_head(out4)
            z = F.normalize(z) #dim=1 normalized
            return p,z
        elif mode == 'p_f':
            #p = self.cls_head_src(out4)
            p= self.cls_head_src(out4)
            return p, out4
        elif mode == 'encoder':
            out4 = F.normalize(out4) #this does not effect accuracy
            return out4
        
    
class ConvNetVis(nn.Module):
    ''' 
    For easy visualization, the feature extractor outputs 2-d features
    '''
    def __init__(self, imdim=3):
        super(ConvNetVis, self).__init__()

        self.conv1 = nn.Conv2d(imdim, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=False) #TODO- inplace=True
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=False) #TODO- inplace=True
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.relu3 = nn.ReLU(inplace=False) #TODO- inplace=True
        self.fc2 = nn.Linear(1024, 2)
        self.relu4 = nn.ReLU(inplace=False) #TODO- inplace=True
        
        self.cls_head_src = nn.Linear(2, 10)
        self.cls_head_tgt = nn.Linear(2, 10)
        self.pro_head = nn.Linear(2, 128)

    def forward(self, x, mode='test'):
        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))
        
        if mode == 'test':
            p = self.cls_head_src(out4)
            return p
        elif mode == 'train':
            p = self.cls_head_src(out4)
            z = self.pro_head(out4)
            z = F.normalize(z)
            return p,z
        elif mode == 'p_f':
            p = self.cls_head_src(out4)
            return p, out4
        
    


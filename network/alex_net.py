
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

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



class ConvNet(nn.Module):
    ''' The network structure is consistent with the SimCLR method
     '''
    def __init__(self, encoder, projection_dim, n_features, output_dim, imdim=3, oracle= False):
        super(ConvNet, self).__init__()
        #added 
        self.encoder= encoder
        self.projection_dim= projection_dim
        self.n_features= n_features
        self.output_dim= output_dim
        #added for miro
        self.oracle= oracle
        self.selected_out = OrderedDict()
        self.fhooks=[]
        self.buffer_features= 7392
        
        
        self.encoder.classifier[-1] = Identity() #model.classifier[6]
        self.encoder_features=  encoder.classifier[1].in_features
        
        self.buffer = nn.Sequential(
            nn.Linear(self.encoder_features, self.buffer_features, bias=False))
        self.encoder.classifier[1]=  nn.Linear(self.buffer_features, self.encoder.classifier[1].out_features)
        self.cls_head_src = nn.Linear(self.n_features, self.output_dim)
        self.pro_head = nn.Sequential(
            nn.Linear(self.buffer_features, self.buffer_features, bias=False),  #self.n_features -> self.projection_dim
            nn.BatchNorm1d(self.buffer_features),
            nn.ReLU(),
            nn.Linear(self.buffer_features, self.buffer_features, bias=False),  #self.n_features -> self.projection_dim
            nn.BatchNorm1d(self.buffer_features),
            nn.ReLU(),
            nn.Linear(self.buffer_features, self.projection_dim, bias=False) #self.n_features,self.projection_dim -> self.projection_dim,self.projection_dim
        )
    
    def get_hook(self):   
        for i,l in enumerate(list(self.encoder._modules.keys())):
            self.fhooks.append(getattr(self.encoder,l).register_forward_hook(self.forward_hook(l)))
        
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
    def freeze_bn(self):
        #for m in self.encoder._modules():
        #    if isinstance(m, nn.BatchNorm2d):
        #        m.eval()
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()
    
        
    def forward(self, x, mode='test'):
        in_size = x.size(0)
        x=self.encoder.features(x)
        x=self.encoder.avgpool(x)
        x=torch.flatten(x,1)
        encoded=self.buffer(x)
        
        
        #encoded= self.encoder(x)
        if mode == 'test':
            cls_encoded= self.encoder.classifier(encoded)
            p= self.cls_head_src(cls_encoded)
            return p
        elif mode == 'train':
            cls_encoded= self.encoder.classifier(encoded)
            p= self.cls_head_src(cls_encoded)
            z = self.pro_head(encoded)
            z = F.normalize(z) #dim=1 normalized
            return p,z
        elif mode == 'encoder':
            encoded = F.normalize(encoded) #this does not effect accuracy
            return encoded
        elif mode == 'encoder_intermediate':
            encoded = F.normalize(encoded) #this does not effect accuracy
            return encoded, self.selected_out
        elif mode == 'prof':
            cls_encoded= self.encoder.classifier(encoded)
            p= self.cls_head_src(cls_encoded) 
            z = self.pro_head(encoded)
            z = F.normalize(z)
            encoded = F.normalize(encoded)
            return p,z,encoded
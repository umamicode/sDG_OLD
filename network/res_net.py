
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
        self.buffer_features= 3024 #3024/7392
        
        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()
        
        self.buffer = nn.Sequential(
            nn.Linear(self.n_features, self.buffer_features, bias=False))
        
        
        self.cls_head_src = nn.Linear(self.buffer_features, self.output_dim)
        
        #self.cls_head_tgt = nn.Linear(self.n_features, self.output_dim)        
        #[TODO]- MLP for Contrastive Learning -Following model design of BarlowTwins Paper
        
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
        
        encoded= self.encoder(x)
        encoded = self.buffer(encoded)
        if mode == 'test':
            p= self.cls_head_src(encoded)
            return p
        elif mode == 'train':
            p= self.cls_head_src(encoded)
            z = self.pro_head(encoded)
            z = F.normalize(z) #dim=1 normalized
            return p,z
        elif mode == 'p_f':
            p= self.cls_head_src(encoded)
            return p, out4
        elif mode == 'encoder':
            encoded = F.normalize(encoded) #this does not effect accuracy
            return encoded
        elif mode == 'encoder_intermediate':
            encoded = F.normalize(encoded) #this does not effect accuracy
            return encoded, self.selected_out
        elif mode == 'prof':
            p= self.cls_head_src(encoded)
            z = self.pro_head(encoded)
            z = F.normalize(z)
            encoded = F.normalize(encoded)
            return p, z, encoded
            
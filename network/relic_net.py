import torch.nn as nn
import torch.nn.functional as F

import torchvision

from network.modules.resnet_hacks import modify_resnet_model
from network.modules.identity import Identity


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, bias= False)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, output_dim, bias= False)

    def forward(self, x):
        return self.fc2(self.relu(self.bn1(self.fc1(x))))

class OnlineNetwork(nn.Module):
    
    def __init__(self, encoder, encoder_dim, projection_dim):
        super(OnlineNetwork, self).__init__()
        self.encoder = encoder 
        self.proj_head = MLP(encoder_dim, projection_dim)
        self.pred_head = MLP(projection_dim, projection_dim)

    def forward(self, x):
        x = self.pred_head(self.proj_head(self.encoder(x)))
        return F.normalize(x, dim=-1, p=2)

class TargetNetwork(nn.Module):

    def __init__(self, encoder, encoder_dim, projection_dim):
        super(TargetNetwork, self).__init__()
        self.encoder = encoder 
        self.proj_head = MLP(encoder_dim, projection_dim)

    def forward(self, x):
        x = self.proj_head(self.encoder(x))
        return F.normalize(x, dim=-1, p=2)

class ReLIC(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(ReLIC, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()
        '''
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim, bias=False),
        )
        '''
        self.online_network= OnlineNetwork(self.encoder, self.n_features, projection_dim)
        self.target_network= TargetNetwork(self.encoder, self.n_features, projection_dim)
        self.cls_head_src = nn.Linear(self.n_features, 10)
        self.cls_head_tgt = nn.Linear(self.n_features, 10)
        
    def forward(self, x_i, x_j, x_orig, test= False):
        
        if test== False:
            online_1, target_1 = self.online_network(x_i), self.target_network(x_i)
            online_2, target_2 = self.online_network(x_j), self.target_network(x_j)
            orig_features = self.online_network(x_orig)


            return online_1,target_1, online_2, target_2, orig_features

        if test==True:
            representation= self.encoder(x_i)
            return representation

        

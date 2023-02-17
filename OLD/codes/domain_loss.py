import torch
import torch.nn as nn
import torch.nn.functional as F



class DomainLoss(nn.Module):
    def __init__(self, projection_dim,device=None):
        super(DomainLoss,self).__init__()
        self.projection_dim= projection_dim
        self.device= device
        self.batch_size= None
        self.norm = nn.InstanceNorm1d(projection_dim, affine=False)
        self.kl_loss= nn.KLDivLoss(reduction="batchmean", log_target=False)
    def forward(self,features):
        if self.device is not None:
            device = self.device
        else:
            device = (torch.device('cuda')
                      if features.is_cuda
                      else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        self.batch_size = features.shape[0]
        contrast_count = features.shape[1]
        anchor_contrast_feature = torch.unbind(features, dim=1)
        anchor_feature= anchor_contrast_feature[0]
        contrast_feature= anchor_contrast_feature[1]
        
        #IN: remove instance specific information
        anchor_feature=torch.unsqueeze(anchor_feature, dim=1) #(N,L)->(N,C,L)
        contrast_feature=torch.unsqueeze(contrast_feature, dim=1) #(N,L)->(N,C,L)

        anchor_feature= self.norm(anchor_feature)
        contrast_feature= self.norm(contrast_feature)
        
        output= 1/ self.kl_loss(anchor_feature,contrast_feature)
        return output
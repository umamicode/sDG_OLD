from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

#[TODO] - add VicReg Loss
class VicReg(nn.Module):
    """Supervised Contrastive Learning with VicReg.
    It also supports the unsupervised contrastive loss in VicReg"""
    def __init__(self, projection_dim,batchsize, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(VicReg, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.projection_dim= projection_dim #[TODO] - added to normalize loss
        self.device=device
        self.batch_size= batchsize
        self.sim_coeff= 25.0
        self.std_coeff= 25.0
        self.cov_coeff= 1.0
        

    def forward(self, features, labels=None, mask=None, adv=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to self-supervised VicReg loss:

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
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

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1] #features.shape= torch.Size([128, 2, 128])
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #unbind to create n views of (k*k) tensors and concat to create (n*k) tensor - (torch.Size([256, 128]))
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] #first k*k --torch.Size([128, 128])
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature # torch.Size([256, 128])
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        #VicReg
        '''
        Reference: https://github.com/facebookresearch/vicreg
        '''
        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of a square matrix
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
        #{TODO}- 1. Normalize representation along the batch dimension
        repr_loss = F.mse_loss(anchor_feature, contrast_feature) #added vicreg
        
        anchor_feature= (anchor_feature - anchor_feature.mean(0)) #/ anchor_feature.std(0) #torch.Size([256, 128])
        contrast_feature = (contrast_feature - contrast_feature.mean(0)) #/ contrast_feature.std(0) #torch.Size([256, 128])
        
        
        
        std_x = torch.sqrt(anchor_feature.var(dim=0) + 0.0001)
        std_y = torch.sqrt(contrast_feature.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        
        cov_x = (anchor_feature.T @ anchor_feature) / (self.batch_size - 1)
        cov_y = (contrast_feature.T @ contrast_feature) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.projection_dim 
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.projection_dim) #self.num_features -> self.projection_dim

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss
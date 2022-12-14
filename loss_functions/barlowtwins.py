from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class BarlowTwinsLoss(nn.Module):
    """Supervised Contrastive Learning with BarlowTwins.
    It also supports the unsupervised contrastive loss in BarlowTwins"""
    def __init__(self, projection_dim, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(BarlowTwinsLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.projection_dim= projection_dim #[TODO] - added to normalize loss
        self.device=device
        #self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, features, labels=None, mask=None, adv=False, oracle = False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to self-supervised BarlowTwins loss:
        https://arxiv.org/pdf/2103.03230v3.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        #Compute Off-Diagonal Elements for Barlow Twins
        def off_diagonal(x):
                # return a flattened view of the off-diagonal elements of a square matrix
                n, m = x.shape
                assert n == m
                return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
            
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
        
        
        
        #ADV-- False torch.Size([512, 128]) torch.Size([512, 128])
        #ADV-- True torch.Size([256, 128]) torch.Size([256, 128])
        
        #BARLOW TWINS
        #Many2Many redundancy reduction 
        #Reference: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        
        contrast_count = features.shape[1]
        anchor_contrast_feature = torch.unbind(features, dim=1)
        
        if adv:
            #Only A Pair of Representations is given
            anchor_feature= anchor_contrast_feature[0]
            contrast_feature= anchor_contrast_feature[1]
            # normalize repr. along the batch dimension
            anchor_feature= (anchor_feature - anchor_feature.mean(0)) / anchor_feature.std(0) #torch.Size([256, 128])
            contrast_feature = (contrast_feature - contrast_feature.mean(0)) / contrast_feature.std(0) #torch.Size([256, 128])
 
            c= torch.matmul(anchor_feature.T, contrast_feature) 
            c.div_(batch_size)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() # appr. 2~3
            off_diag = off_diagonal(c).pow_(2).sum()
            
            loss = on_diag + 0.051 * off_diag
            loss = -1*loss # Changed from loss = 1/loss. See BT Run1 (Better than Run0)#og
            #loss= (1/loss)**2 #run0
        if not adv:
            #Given 4 Representations 
            total_loss= 0.0
            #scenarios= list(itertools.combinations(list(range(len(anchor_contrast_feature))), 2)) #for faster computation
            for p,anchor_feature in enumerate(anchor_contrast_feature):
                for q,contrast_feature in enumerate(anchor_contrast_feature):
                    if p != q: #og
                    #if (p,q) in scenarios: #for faster computation
                        # normalize repr. along the batch dimension
                        if not oracle:
                            anchor_feature= (anchor_feature - anchor_feature.mean(0)) / anchor_feature.std(0) #torch.Size([256, 128])
                            contrast_feature = (contrast_feature - contrast_feature.mean(0)) / contrast_feature.std(0) #torch.Size([256, 128])
                        
                        c= torch.matmul(anchor_feature.T, contrast_feature) 
                        c.div_(batch_size)
                        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() # appr. 2~3
                        off_diag = off_diagonal(c).pow_(2).sum()
                        
                        loss = on_diag + 0.0051 * off_diag
                        total_loss += loss
            loss = total_loss / (len(anchor_contrast_feature)**2) #og
            #loss= total_loss / len(scenarios) #for faster computation
            
        return loss
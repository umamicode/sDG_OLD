from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class BarlowTwinsLoss(nn.Module):
    """Supervised Contrastive Learning with BarlowTwins for the Oracle.
    It also supports the unsupervised contrastive loss in BarlowTwins"""
    def __init__(self, projection_dim, lmda=0.051, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(BarlowTwinsLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.projection_dim= projection_dim #[TODO] - added to normalize loss
        self.penalty= projection_dim/128 #normalized loss
        self.device=device
        self.lmda= lmda
        #self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, features, labels=None, mask=None, adv=False, standardize = True):
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
            anchor_feature= (anchor_feature - anchor_feature.mean(0)) / anchor_feature.std(0) 
            contrast_feature = (contrast_feature - contrast_feature.mean(0)) / contrast_feature.std(0)
 
            c= torch.matmul(anchor_feature.T, contrast_feature) 
            c.div_(batch_size)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()# / (self.penalty)
            off_diag = off_diagonal(c).pow_(2).sum()# / (self.penalty * (self.penalty -1))
            if self.projection_dim != 128:
                on_diag /= (self.penalty)
                off_diag /= (self.penalty * (self.penalty -1))
            #OG ADV LOSS (NE PAS TOUCHER) -lmda test results: 0.051 optimal
            loss = on_diag + self.lmda * off_diag #satur/run0
            #loss = -1*loss
            
            
            #Candidates
            #simple ADV LOSS (default:lmda=0.1)
            #loss= on_diag / (off_diag * self.lmda)
            #others
            #loss = 1/ (loss + 1e-6) #loss_exp
            #loss = 0.1*on_diag + torch.exp(1 / (0.051* off_diag + 1e-6))
            #loss = on_diag / (loss +1e-6) #satur/run2
             
        if not adv:
            #Given 4 Representations 
            total_loss= 0.0
            #scenarios= list(itertools.combinations(list(range(len(anchor_contrast_feature))), 2)) #for faster computation
            for p,anchor_feature in enumerate(anchor_contrast_feature):
                for q,contrast_feature in enumerate(anchor_contrast_feature):
                    if p != q: #og
                        # normalize repr. along the batch dimension
                        if standardize:
                            #standardization (added 1e-6 so that nan does not appear)
                            anchor_feature= (anchor_feature - anchor_feature.mean(0)) / (anchor_feature.std(0)+ 1e-6) #torch.Size([256, 128])
                            contrast_feature = (contrast_feature - contrast_feature.mean(0)) / (contrast_feature.std(0) +1e-6)  #torch
                        if not standardize:
                            pass
                        
                        c= torch.matmul(anchor_feature.T, contrast_feature) 
                        c.div_(batch_size)
                        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()# / self.penalty
                        off_diag = off_diagonal(c).pow_(2).sum()# / (self.penalty * (self.penalty -1))
                        if self.projection_dim != 128:
                            on_diag /= (self.penalty)
                            off_diag /= (self.penalty * (self.penalty -1))
                        loss = on_diag + 0.0051 * off_diag
                        total_loss += loss
            loss = total_loss / (len(anchor_contrast_feature)**2) #og
            #works better than dividing with (N^2 -N)
            
            
        return loss
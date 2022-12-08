from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Reference Code: https://github.com/lileicv/PDEN 
    Original Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device=device

    def forward(self, features, labels=None, mask=None, adv=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

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

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #unbind to create n views of (k*k) tensors and concat to create (n*k) tensor
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] #first k*k
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits (shape: torch.Size([256, 256])
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        #log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        if adv:
            log_prob = torch.log( 1- exp_logits / (exp_logits.sum(1, keepdim=True)+1e-6) - 1e-6)
        else:
            log_prob = torch.log( exp_logits / (exp_logits.sum(1, keepdim=True)+1e-6) +1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


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

    def forward(self, features, labels=None, mask=None, adv=False):
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
        
        
        
        #FIXED ANCHOR/CONTRAST FOR BARLOWTWINS 
        contrast_count = features.shape[1]
        anchor_contrast_feature = torch.unbind(features, dim=1)
        anchor_feature= anchor_contrast_feature[0]
        contrast_feature= anchor_contrast_feature[1]
            
            
        #BARLOW TWINS
        '''
        Reference: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        '''
        #{TODO}- 1. Normalize representation along the batch dimension
        anchor_feature= (anchor_feature - anchor_feature.mean(0)) / anchor_feature.std(0) #torch.Size([256, 128])
        contrast_feature = (contrast_feature - contrast_feature.mean(0)) / contrast_feature.std(0) #torch.Size([256, 128])
            
        #12/08/22 Test
            
            
        #{TODO}- 2. MatMul for cross-correlation matrix
        #c= torch.matmul(anchor_feature, contrast_feature.T) 
        c= torch.matmul(anchor_feature.T, contrast_feature) # (N*D).T @ (N*D) -> (D*D)
        c.div_(batch_size) #c --torch.Size([256, 256])
            
        #{TODO}- 3. Loss    
        #[TODO] ADV: Maximize Loss
        if adv:
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() # appr. 2~3
            #off_diag = off_diagonal(c.add_(-1)).pow_(2).sum()  # appr. 1700~2000
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + 0.0051 * off_diag
        #Non-ADV: Minimize Loss
        else: 
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + 0.0051 * off_diag  #lambda=0.0051 suggested in barlowtwins paper
        
        return loss

class PRISMLoss(nn.Module):
    """Supervised Contrastive Learning with PRISM.
    It also supports the unsupervised contrastive loss in PRISM"""
    def __init__(self, projection_dim, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(PRISMLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.projection_dim= projection_dim #[TODO] - added to normalize loss
        self.device=device
        #self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, features, labels=None, mask=None, adv=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to self-supervised PRISM loss:
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
        
        
        if adv:
            
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
            # compute logits (shape: torch.Size([256, 256])
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()


            # tile mask
            mask = mask.repeat(anchor_count, contrast_count)
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask
            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = torch.log(1- exp_logits / (exp_logits.sum(1, keepdim=True)+1e-6) - 1e-6)
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, batch_size).mean()
            
            
            
        
        else:
            #FIXED ANCHOR/CONTRAST FOR BARLOWTWINS 
            contrast_count = features.shape[1]
            anchor_contrast_feature = torch.unbind(features, dim=1)
            anchor_feature= anchor_contrast_feature[0]
            contrast_feature= anchor_contrast_feature[1]
            
            
            #BARLOW TWINS
            '''
            Reference: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
            '''
            #{TODO}- 1. Normalize representation along the batch dimension
            anchor_feature= (anchor_feature - anchor_feature.mean(0)) / anchor_feature.std(0) #torch.Size([256, 128])
            contrast_feature = (contrast_feature - contrast_feature.mean(0)) / contrast_feature.std(0) #torch.Size([256, 128])
            
            #12/08/22 Test
            
            
            #{TODO}- 2. MatMul for cross-correlation matrix
            #c= torch.matmul(anchor_feature, contrast_feature.T) 
            c= torch.matmul(anchor_feature.T, contrast_feature) # (N*D).T @ (N*D) -> (D*D)
            c.div_(batch_size) #c --torch.Size([256, 256])
            
            #{TODO}- 3. Loss    
            #[TODO] ADV: Maximize Loss
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + 0.0051 * off_diag  #lambda=0.0051 suggested in barlowtwins paper
            
            
            
            
        return loss



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
    

if __name__=='__main__':
    import torch.nn.functional as F
    torch.manual_seed(0)
    x = torch.randn(32, 2, 10)
    x = F.normalize(x)
    y = torch.randint(0, 10, [32])
    loss_layer = SupConLoss()
    loss = loss_layer(x, y)
    print(loss)


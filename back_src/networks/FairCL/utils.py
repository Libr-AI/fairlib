import torch
import torch.nn as nn

class Contrastive_Loss(torch.nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    
    From https://github.com/AiliAili/contrastive_learning_fair_representations/blob/master/networks/contrastive_loss.py
    """

    def __init__(self, device, temperature=0.07, base_temperature=0.07):
        super(Contrastive_Loss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        assert labels.shape[0] == batch_size
        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = 1
        contrast_feature = features

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        #compute_logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        
        #for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast-logits_max.detach()
        
        #tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        #mask out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size*anchor_count).view(-1, 1).to(self.device), 0)
        
        mask = mask*logits_mask

        #compute log prob
        exp_logits = torch.exp(logits)*logits_mask+1e-20

        
        log_prob = logits-torch.log(exp_logits.sum(1, keepdim=True))

        #compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask*log_prob).sum(1)/(1+mask.sum(1))

        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss
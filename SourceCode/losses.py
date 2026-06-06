
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


def cross_entropy(output, target):
    return torch.sum(-target * F.log_softmax(output, dim=1), dim=1).mean()


def get_criterion(reduction='mean'):
    return nn.CrossEntropyLoss(reduction=reduction)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class L1L2RegularizationLoss(nn.Module):
    def __init__(self, model, reg_type='l2', factor=5e-4):
        super(L1L2RegularizationLoss, self).__init__()
        self.model = model
        self.reg_type = reg_type.lower()
        self.factor = factor
    
    def forward(self):
        if self.reg_type == 'l2':
            reg_loss = 0
            for param in self.model.parameters():
                reg_loss += torch.sum(param ** 2)
            return self.factor * reg_loss
        elif self.reg_type == 'l1':
            reg_loss = 0
            for param in self.model.parameters():
                reg_loss += torch.sum(torch.abs(param))
            return self.factor * reg_loss
        else:
            return 0


CrossEntropy = get_criterion

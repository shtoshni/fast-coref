import torch.nn as nn
import torch


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target, weight=None):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.shape[self.dim] - 1))
            true_dist.scatter_(self.dim, target, self.confidence)

        if weight is None:
            loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        else:
            loss = torch.sum(-true_dist * pred * weight, dim=self.dim) / torch.sum(weight)

        return loss

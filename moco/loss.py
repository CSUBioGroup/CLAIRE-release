import torch
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-8

def entropy(x):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

class SCANLoss(nn.Module):
    def __init__(self, lamda1):
        super().__init__()
        self.lamda1 = lamda1

        self.bce = nn.BCELoss()

    def forward(self, similarity, target, soft_assignment):
        consistency_loss = self.bce(similarity, target)  # Log<similarity>

        # clustering loss, part 2, entropy loss
        entr_loss = entropy(torch.mean(soft_assignment, dim=0))   # batch-wise entropy
        # entr = entropy(anchor_scores)  # sample-wise entropy 

        loss = consistency_loss - self.lamda1 * entr_loss


        return loss, consistency_loss, entr_loss











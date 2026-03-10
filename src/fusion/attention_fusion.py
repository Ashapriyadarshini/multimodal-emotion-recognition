import torch
import torch.nn as nn


class AttentionFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.attention = nn.Linear(256,1)

    def forward(self,v,a,t):

        modalities = torch.stack([v,a,t],dim=1)

        weights = torch.softmax(self.attention(modalities),dim=1)

        fused = torch.sum(weights*modalities,dim=1)

        return fused
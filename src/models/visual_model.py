import torch
import torch.nn as nn
import torchvision.models as models
from modules.se_block import SEBlock


class VisualModel(nn.Module):

    def __init__(self):
        super().__init__()

        vgg = models.vgg16(pretrained=True)

        self.features = vgg.features

        self.se = SEBlock(512)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(512,256)

    def forward(self,x):

        x = self.features(x)

        x = self.se(x)

        x = self.pool(x)

        x = x.view(x.size(0),-1)

        x = torch.relu(self.fc(x))

        return x
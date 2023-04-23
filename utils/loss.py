import torch
import torch.nn as nn
from torchvision.models import vgg16


class AdversialLoss(nn.Module):
    def __init__(self):
        super(AdversialLoss, self).__init__()

    def forward(self, target, style, edge=None,eps=1e-10):
        if edge is None:
            return -(torch.log(1 - target).mean() +\
                     torch.log(style + eps).mean())
        else:
            return -(torch.log(style+eps).mean() +\
                     torch.log(1-edge).mean() +\
                     torch.log(1-target).mean())
        

class ContentLoss(nn.Module):
    def __init__(self, omega=10):
        super(ContentLoss, self).__init__()

        self.base_loss = nn.L1Loss()
        self.omega = omega

        perception = list(vgg16(pretrained=True).features)[:25]
        self.perception = nn.Sequential(*perception).eval()

        for param in self.perception.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        x1 = self.perception(x1)
        x2 = self.perception(x2)
        
        return self.omega * self.base_loss(x1, x2)
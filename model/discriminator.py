import torch
from torch import nn
from ._layer import DownConvBlock

class Discriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            DownConvBlock(3,32,kernel_size=3,stride=1,use_norm=False), # img_size 

            nn.ReflectionPad2d(1),
            DownConvBlock(32,64,kernel_size=3,stride=2,use_norm=False), # img_size / 2
            nn.ReflectionPad2d(1),
            DownConvBlock(64,128,kernel_size=3,stride=1), # img_size / 2

            nn.ReflectionPad2d(1),
            DownConvBlock(128,128,kernel_size=3,stride=2,use_norm=False), # img_size / 4
            nn.ReflectionPad2d(1),
            DownConvBlock(128,256,kernel_size=3,stride=1), # img_size / 4

            nn.ReflectionPad2d(1),
            DownConvBlock(256,256,kernel_size=3,stride=1), # img_size / 4

            nn.ReflectionPad2d(1),
            DownConvBlock(256,1,kernel_size=3,stride=1,use_act=False,use_norm=False),

            nn.Sigmoid()
        )

    def forward(self,x):
        return self.conv(x)
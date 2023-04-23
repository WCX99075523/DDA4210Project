import torch
from torch import nn
from ._layer import DownConvBlock, ResBlock, UpConvBlock

class Generator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv_down = nn.Sequential(
            # Starting with Large kernel size at first
            DownConvBlock(3,64,kernel_size=7,stride=1,padding=7//2,negative_slope=0), # img of original size

            nn.ReflectionPad2d(1),
            DownConvBlock(64,128,kernel_size=3,stride=2,use_act=False,use_norm=False,negative_slope=0), # img_size / 2
            nn.ReflectionPad2d(1),
            DownConvBlock(128,128,kernel_size=3,stride=1,negative_slope=0), # img_size / 2

            nn.ReflectionPad2d(1),
            DownConvBlock(128,256,kernel_size=3,stride=2,use_act=False,use_norm=False,negative_slope=0), # img_size / 4
            nn.ReflectionPad2d(1),
            DownConvBlock(256,256,kernel_size=3,stride=1,negative_slope=0) # img_size / 4
        )
        self.res_bloks = nn.Sequential(
            *[ResBlock(256,negative_slope=0)] * 8    # img_size / 4
        )
        self.conv_up = nn.Sequential(
            UpConvBlock(256,128,kernel_size=3,stride=2,use_avgpool=True,use_act=False,use_norm=False), # img_size / 2
            nn.ReflectionPad2d(1),
            DownConvBlock(128,128,kernel_size=3,stride=1,negative_slope=0), # img_size / 2

            UpConvBlock(128,64,kernel_size=3,stride=2,use_avgpool=True,use_act=False,use_norm=False),
            nn.ReflectionPad2d(1),
            DownConvBlock(64,64,kernel_size=3,stride=1,negative_slope=0), # img_size

            # UpConvBlock(64,64,kernel_size=3,stride=2,use_avgpool=True,use_act=False,use_norm=False),
            # nn.ReflectionPad2d(1),
            DownConvBlock(64,3,kernel_size=7,stride=1,padding=7//2,use_act=False,use_norm=False,negative_slope=0), # img_size 
        )

    def forward(self,x):
        x = self.conv_down(x)
        x = self.res_bloks(x)
        return self.conv_up(x)

class UnetGenerator(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        pass

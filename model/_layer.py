import torch 
from torch import nn

class DownConvBlock(nn.Module):

    def __init__(
            self,
            # Covolutional parameters
            in_channels: int, out_channels: int, 
            stride: int=1, kernel_size: int=3, padding: int=0,
            # norm, activation parameters and max_pooling
            use_norm: bool=True,
            use_act: bool=True,negative_slope: float=0.2,
            use_avgpool: bool=False
            ) -> None:
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False))
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_act:
            layers.append(nn.LeakyReLU(negative_slope))
        if use_avgpool:
            layers.append(nn.AvgPool2d(2,1))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)
    
class ResBlock(nn.Module):

    def __init__(self, n_channels: int, negative_slope: float=0.2) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            DownConvBlock(n_channels,n_channels),
            nn.ReflectionPad2d(1),
            DownConvBlock(n_channels,n_channels,use_act=False)
        )
        # self.act = nn.LeakyReLU(negative_slope)

    def forward(self,x):
        r = x
        x = self.conv_block(x)
        x += r
        return x
        # return self.act(x)

class UpConvBlock(nn.Module):

    def __init__(
            self,
            # Convtranspose parameters
            in_channels: int, out_channels: int, 
            kernel_size: int=3, stride: int=2, padding: int=0,
            # norm, activation parameters and max_pooling
            use_norm: bool=True,
            use_act: bool=True, negative_slope: float=0.2,
            use_avgpool: bool=False
        ) -> None:
        super().__init__()
        layers = []
        layers.append(
            nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
            )
        )
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_act:
            layers.append(nn.LeakyReLU(negative_slope))
        if use_avgpool:
            layers.append(nn.AvgPool2d(2,1))
        self.conv_block = nn.Sequential(*layers)

    def forward(self,x):
        return self.conv_block(x)
    
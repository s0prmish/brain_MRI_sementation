import torch,cv2


import torch.autograd.profiler as profiler
import torch.nn.functional as F


from torch import nn
from torch.nn import Module
from torch.optim import RMSprop
from data.dataloader import MriSegmentation


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, inner=False, stride=1, padding=0):
        super(DoubleConv, self).__init__()

        if inner == True:
            self.two_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.two_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

    def forward(self, x):
        # print(self.two_conv(x).shape)
        return self.two_conv(x)


class UpandAdd(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=False, stride=1, padding=0):
        super(UpandAdd, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DoubleConv(in_channels, out_channels, True)
            )

        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
                DoubleConv(in_channels, out_channels, True)
            )

    def forward(self, x):
        x = self.up(x)
        return x


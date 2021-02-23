import torch,cv2


import torch.autograd.profiler as profiler
import torch.nn.functional as F


from torch import nn
from torch.nn import Module
from torch.optim import RMSprop
from data.dataloader import MriSegmentation
from model_parts import DoubleConv,UpandAdd




class Net(nn.Module):

    def __init__(self, in_channels, n_classes=2, bilinear=True, inner=False):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            DoubleConv(in_channels, 64, False),
            DoubleConv(64, 128, False),
            DoubleConv(128, 256, False),
            DoubleConv(256, 512, False),
            DoubleConv(512, 1024, True),
            UpandAdd(1024, 512, True),
            UpandAdd(512, 256, True),
            UpandAdd(256, 128, True),
            UpandAdd(128, 64, True),
        )
        self.classification = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        return self.classification(x)


# sample_net=UpandAdd(1024,512)
# x=torch.randn(1,1024,28,28)
# sample_net(x).shape

# sample=Net(3,2,bilinear=True,inner=False)
# ex=torch.randn(1,3,256,256)
# sample.eval()
# sample(ex).shape

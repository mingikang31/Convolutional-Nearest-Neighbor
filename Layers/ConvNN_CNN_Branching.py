'''Branching Layer with Conv2d and ConvNN 2D layers'''

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary

# ConvNN
from Conv1d_NN import *
from Conv2d_NN import *

from Conv1d_NN_spatial import *
from Conv2d_NN_spatial import *

from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D


class ConvNN_CNN_Random_BranchingLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, K=9, samples = "all", location_channels = False):
        super(ConvNN_CNN_Random_BranchingLayer, self).__init__()
        self.kernel_size = kernel_size
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            Conv2d_NN(in_ch, out_ch, K = K, stride = K, samples = samples, location_channels = location_channels), 
            nn.ReLU()
        )
        self.reduce_channels = nn.Conv2d(out_ch*2, out_ch, 1)


    def forward(self, x):
        
        x1 = self.branch1(x)
        
        x2 = self.branch2(x)
        
        concat = torch.cat([x1, x2], dim=1)
        
        reduce = self.reduce_channels(concat)
        return reduce

class ConvNN_CNN_Spatial_BranchingLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, K=9, samples = 8, location_channels = False):
        super(ConvNN_CNN_Spatial_BranchingLayer, self).__init__()
        self.kernel_size = kernel_size
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            Conv2d_NN_spatial(in_ch, out_ch, K = K, stride = K, samples = samples, location_channels = location_channels), 
            nn.ReLU()
        )
        self.reduce_channels = nn.Conv2d(out_ch*2, out_ch, 1)

    def forward(self, x):
        
        x1 = self.branch1(x)
        
        x2 = self.branch2(x)
        
        concat = torch.cat([x1, x2], dim=1)
        
        reduce = self.reduce_channels(concat)
        return reduce


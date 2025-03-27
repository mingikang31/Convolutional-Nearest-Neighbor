'''Branching Layer with Conv2d and ConvNN 2D layers'''

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary
import numpy as np

# ConvNN
from Conv1d_NN import *
from Conv2d_NN import *

from Conv1d_NN_spatial import *
from Conv2d_NN_spatial import *

from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D


class ConvNN_CNN_Random_BranchingLayer(nn.Module):
    def __init__(self, in_ch, out_ch, channel_ratio=(16, 16), kernel_size=3, K=9, samples = "all", location_channels = False):
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_ch, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(ConvNN_CNN_Random_BranchingLayer, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        self.samples = samples
        self.location_channels = location_channels
        
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, channel_ratio[0], kernel_size, stride=1, padding=1),
                nn.ReLU()
            )
        
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN(in_ch, channel_ratio[1], K = K, stride = K, samples = samples, location_channels = location_channels), 
                nn.ReLU()
            )
        

        self.reduce_channels = nn.Conv2d(out_ch*2, out_ch, 1)


    def forward(self, x):
        
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
        
            concat = torch.cat([x1, x2], dim=1)
        
        reduce = self.reduce_channels(concat)
        return reduce

class ConvNN_CNN_Spatial_BranchingLayer(nn.Module):
    def __init__(self, in_ch, out_ch, channel_ratio=(16, 16), kernel_size=3, K=9, samples = 8, location_channels = False):
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_ch, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(ConvNN_CNN_Spatial_BranchingLayer, self).__init__()
        self.kernel_size = kernel_size
        
        self.in_ch = in_ch 
        self.out_ch = out_ch    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        self.samples = samples
        self.location_channels = location_channels
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, channel_ratio[0], kernel_size, stride=1, padding=1),
                nn.ReLU()
            )
            
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_spatial(in_ch, channel_ratio[1], K = K, stride = K, samples = samples, location_channels = location_channels), 
                nn.ReLU()
            )

        
        self.reduce_channels = nn.Conv2d(out_ch*2, out_ch, 1)

    def forward(self, x):
        
        if self.channel_ratio[0] != 0:
            x1 = self.branch1(x)
        
        if self.channel_ratio[1] != 0:
            x2 = self.branch2(x)
        
        if self.channel_ratio[0] == 0:
            concat = x2
        elif self.channel_ratio[1] == 0:
            concat = x1
        else:
        
            concat = torch.cat([x1, x2], dim=1)
        
        reduce = self.reduce_channels(concat)
        return reduce
    
'''EXAMPLE USAGE'''
def example_usage():
    '''Example Usage of ConvNN_CNN_Random_BranchingLayer and ConvNN_CNN_Spatial_BranchingLayer'''
    ex = torch.rand(32, 3, 28, 28)
    print("Input: ", ex.shape)
    
    convnn_cnn_random = ConvNN_CNN_Random_BranchingLayer(in_ch=3, out_ch=16, channel_ratio=(28, 4), kernel_size=3, K=3, samples=5, location_channels=True)
    
    output_random = convnn_cnn_random(ex)
    print("Output Random Branching: ", output_random.shape) 
    
    convnn_cnn_spatial = ConvNN_CNN_Spatial_BranchingLayer(in_ch=3, out_ch=16, channel_ratio=(28, 4), kernel_size=3, K=3, samples=5, location_channels=True)
    
    output_spatial = convnn_cnn_spatial(ex)
    print("Output Spatial Branching: ", output_spatial.shape)
    
# example_usage()
    

'''Branching Layer with Attention and ConvNN 2D layers'''

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

from Conv1d_NN_Attn import *
from Conv2d_NN_Attn import *

from Conv1d_NN_Attn_V import * 
from Conv2d_NN_Attn_V import *

from Attention1d import Attention1d
from Attention2d import Attention2d

from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D


class Attention_ConvNN_Random_Branching(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 channel_ratio=(16, 16), 
                 K=9, 
                 shuffle_pattern="NA",
                 shuffle_scale=1,
                 num_heads=1, 
                 samples = "all", 
                 location_channels = False):
        
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_ch, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(Attention_ConvNN_Random_Branching, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch    
        self.channel_ratio = channel_ratio
        
        self.K = K
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        self.num_heads = num_heads  
        self.samples = samples
        self.location_channels = location_channels
        
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(self.in_ch,
                            self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
        
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN(self.in_ch, 
                          self.channel_ratio[1], 
                          K=self.K, 
                          stride=self.K, 
                          samples=self.samples, 
                          shuffle_pattern=self.shuffle_pattern,
                          shuffle_scale=self.shuffle_scale,
                          location_channels=self.location_channels), 
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

class Attention_ConvNN_Spatial_Branching(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 channel_ratio=(16, 16), 
                 K=9, 
                 samples = 8, 
                 shuffle_pattern="NA",  
                 shuffle_scale=1,    
                 num_heads=1,   
                 location_channels = False):
        
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_ch, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(Attention_ConvNN_Spatial_Branching, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch    
        self.channel_ratio = channel_ratio
        self.K = K
        self.samples = samples
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_heads = num_heads
        
        self.location_channels = location_channels
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_ch, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_spatial(in_channels=in_ch, 
                                  out_channels=channel_ratio[1], 
                                  K=self.K, 
                                  stride=self.K, 
                                  samples=self.samples, 
                                  shuffle_pattern=self.shuffle_pattern,
                                  shuffle_scale=self.shuffle_scale,
                                  location_channels=self.location_channels), 
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
        
class Attention_ConvNN_Attn_Branching(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 channel_ratio=(16, 16), 
                 kernel_size=3, 
                 K=9, 
                 samples="all", 
                 shuffle_pattern="NA", 
                 shuffle_scale=1, 
                 num_heads=1, 
                 location_channels = False, 
                 image_size=(32, 32)):
        
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_ch, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(Attention_ConvNN_Attn_Branching, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        self.samples = samples
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_heads = num_heads
        self.image_size = image_size
        
        self.location_channels = location_channels
        
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_ch, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn(self.in_ch, 
                               self.channel_ratio[1], 
                               K=self.K, 
                               stride=self.K, 
                               samples=self.samples, 
                               shuffle_pattern=self.shuffle_pattern,
                               shuffle_scale=self.shuffle_scale,
                               image_size=self.image_size, 
                               location_channels=self.location_channels), 
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
    
class Attention_ConvNN_Attn_V_Branching(nn.Module):
    def __init__(self, 
                    in_ch, 
                    out_ch, 
                    channel_ratio=(16, 16), 
                    kernel_size=3, 
                    K=9, 
                    samples="all", 
                    shuffle_pattern="NA", 
                    shuffle_scale=1, 
                    num_heads=1, 
                    location_channels = False, 
                    image_size = (32, 32)):
        
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_ch, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(Attention_ConvNN_Attn_V_Branching, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        self.samples = samples
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_heads = num_heads
        self.image_size = image_size
        
        self.location_channels = location_channels
        
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_ch, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn_V(self.in_ch, 
                               self.channel_ratio[1], 
                               K=self.K, 
                               stride=self.K, 
                               samples=self.samples, 
                               shuffle_pattern=self.shuffle_pattern,
                               shuffle_scale=self.shuffle_scale,
                               image_size=self.image_size, 
                               location_channels=self.location_channels), 
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
    
class Attention_Conv2d_Branching(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 channel_ratio=(16, 16), 
                 kernel_size=3,
                 shuffle_pattern="NA", 
                 shuffle_scale=1, 
                 num_heads=1, 
                 location_channels = False,  
                 ):
        # Channel_ratio must add up to 2*out_ch

        super(Attention_Conv2d_Branching, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_heads = num_heads
        self.location_channels = location_channels

    
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_ch, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        

        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                nn.Conv2d(self.in_ch, 
                          self.channel_ratio[1], 
                          self.kernel_size, 
                          stride=1, 
                          padding=1
                         ),
                nn.ReLU()
            )
        
        self.reduce_channels = nn.Conv2d(out_ch*2, out_ch, 1)

    def forward(self, x):
        x1 = self.branch1(x)

        x2 = self.branch2(x)
        
        concat = torch.cat([x1, x2], dim=1)
        
        reduce = self.reduce_channels(concat)
        return reduce
    
if __name__ == "__main__":
    ex = torch.randn(32, 3, 32, 32)
    print("Input: ", ex.shape)
    
    # Attention + ConvNN Random Branching 
    attention_convnn_random_branching = Attention_ConvNN_Random_Branching(in_ch=3, out_ch=16, channel_ratio=(16, 16), K=9, shuffle_pattern="BA", shuffle_scale=2, num_heads=4, samples="all", location_channels=False)
    output_random = attention_convnn_random_branching(ex)
    print("Output Random Branching: ", output_random.shape)
    
    # Attention + ConvNN Spatial Branching
    attention_convnn_spatial_branching = Attention_ConvNN_Spatial_Branching(in_ch=3, out_ch=16, channel_ratio=(16, 16), K=3, samples=64, shuffle_pattern="NA", shuffle_scale=1, num_heads=4, location_channels=False)
    
    output_spatial = attention_convnn_spatial_branching(ex)
    print("Output Spatial Branching: ", output_spatial.shape)
    
    # Attention + ConvNN Attention Branching
    attention_convnn_attn_branching = Attention_ConvNN_Attn_Branching(in_ch=3, out_ch=16, channel_ratio=(16, 16), kernel_size=3, K=3, samples=64, shuffle_pattern="NA", shuffle_scale=1, num_heads=1, location_channels=False)
    output_attention = attention_convnn_attn_branching(ex)
    print("Output Attention Branching: ", output_attention.shape)
    
    # Attention + ConvNN Attention V Branching
    attention_convnn_attn_v_branching = Attention_ConvNN_Attn_V_Branching(in_ch=3, out_ch=16, channel_ratio=(16, 16), kernel_size=3, K=9, samples=64, shuffle_pattern="NA", shuffle_scale=1, num_heads=1, location_channels=False)
    output_attention_v = attention_convnn_attn_v_branching(ex)
    print("Output Attention V Branching: ", output_attention_v.shape)
    
    # Attention + Conv2d Branching
    attention_conv2d_branching = Attention_Conv2d_Branching(in_ch=3, out_ch=16, channel_ratio=(8, 8), kernel_size=3, shuffle_pattern="BA", shuffle_scale=2, num_heads=1, location_channels=False)
    
    
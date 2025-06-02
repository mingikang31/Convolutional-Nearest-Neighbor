'''Attention 1D Layer'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D

class Attention1d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 shuffle_pattern='N/A', 
                 shuffle_scale=1, 
                 num_heads=1
                 ):
        super(Attention1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_heads = num_heads
        
        
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        
        # Channels for Attention 
        self.in_channels = self.in_channels * self.shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = self.out_channels * self.shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels
        
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.out_channels, num_heads=self.num_heads, batch_first=True)
        
        self.conv1x1 = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
        
    def forward(self, x):
        if self.shuffle_pattern in ["BA", "B"]:
            x1 = self.unshuffle_layer(x)
        else: 
            x1 = x 
        
        x1 = self.conv1x1(x1) # [B, C, N]
        x1 = x1.permute(0, 2, 1)
        
        x2 = self.multi_head_attention(x1, x1, x1)[0] # (B, N, C)
        x2 = x2.permute(0, 2, 1) # (B, C, N)
        
        if self.shuffle_pattern in ["BA", "A"]:
            x3 = self.shuffle_layer(x2)
        else: 
            x3 = x2
        return x3
    
if __name__ == "__main__":
    x = torch.randn(64, 3, 256) 
    
    attention_layer = Attention1d(in_channels=3, out_channels=8, shuffle_pattern='BA', shuffle_scale=2, num_heads=1)
    output = attention_layer(x)
    
    print("Input shape:", x.shape) # Should be (64, 3, 256)
    print("Output shape:", output.shape) # Should be (64, 8, 256)
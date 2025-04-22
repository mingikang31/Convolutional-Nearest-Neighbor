'''Attention 2D Layer'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from Attention1d import Attention1d

class Attention2d(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 shuffle_pattern='N/A',
                 shuffle_scale=1,
                 num_heads=1,
                 location_channels=False,
                 ): 
        super(Attention2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_heads = num_heads
        self.location_channels = location_channels
        
        if (self.shuffle_pattern in ["B", "BA"]):
            if self.location_channels: 
                self.in_channels_1d = (self.in_channels + 2) * (self.shuffle_scale**2)
                self.out_channels_1d = (self.out_channels + 2) * (self.shuffle_scale **2)
            else:
                self.in_channels_1d = self.in_channels * (self.shuffle_scale**2)
                self.out_channels_1d = self.out_channels * (self.shuffle_scale **2)

        else: 
            if self.location_channels: 
                self.in_channels_1d = self.in_channels + 2
                self.out_channels_1d = self.out_channels + 2
            else:
                self.in_channels_1d = self.in_channels
                self.out_channels_1d = self.out_channels
                
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        
        self.attention1d = Attention1d(in_channels=self.in_channels_1d,
                                        out_channels=self.out_channels_1d,
                                        shuffle_pattern="N/A",
                                        shuffle_scale=1,
                                        num_heads=self.num_heads
                                          )
        
        self.flatten = nn.Flatten(start_dim=2)
        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
    def forward(self, x):
        if self.shuffle_pattern in ["B", "BA"]:
            if self.location_channels: 
                x_coordinates = self.coordinate_channels(x.shape, device=x.device)
                x = torch.cat((x, x_coordinates), dim=1)
                x1 = self.unshuffle_layer(x)
            else: 
                x1 = self.unshuffle_layer(x)
            
        else: 
            if self.location_channels:
                x_coordinates = self.coordinate_channels(x.shape, device=x.device)
                x1 = torch.cat((x, x_coordinates), dim=1)
            else: 
                x1 = x
                
        x2 = self.flatten(x1)
        x3 = self.attention1d(x2)

        unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
        x4 = unflatten(x3)

        if self.shuffle_pattern in ["A", "BA"]:
            if self.location_channels:
                x4 = self.shuffle_layer(x4)
                x5 = self.pointwise_conv(x4) ## Added Pointwise Conv to reduce channels added for spatial coordinates
            else:
                x5 = self.shuffle_layer(x4)
        else: 
            if self.location_channels:
                x5 = self.pointwise_conv(x4) ## Added Pointwise Conv to reduce channels added for spatial coordinates
            else: 
                x5 = x4
        return x5
    
    def coordinate_channels(self, tensor_shape, device):
        x_ind = torch.arange(0, tensor_shape[2])
        y_ind = torch.arange(0, tensor_shape[3])
        
        x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
        x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
        xy_grid = torch.cat((x_grid, y_grid), dim=1)
        xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
        return xy_grid_normalized.to(device)

if __name__ == "__main__":
    x = torch.randn(32, 3, 32, 32) 
    attention_layer = Attention2d(in_channels=3, out_channels=8, shuffle_pattern='BA', shuffle_scale=2, num_heads=1)
    output = attention_layer(x)
    
    print("Input shape:", x.shape) # Should be (64, 3, 32, 32)
    print("Output shape:", output.shape) # Should be (64, 8, 32, 32)
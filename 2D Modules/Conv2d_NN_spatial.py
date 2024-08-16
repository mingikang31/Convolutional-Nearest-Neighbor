'''Convolution 2D Spatial Nearest Neighbor Layer'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
from Conv1d_NN_spatial import Conv1d_NN_spatial
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D
import faiss



class Conv2d_NN_spatial(nn.Module): 
   def __init__(self, in_channels, out_channels, K=3, stride=3, padding=0, shuffle_pattern="N/A", shuffle_scale=2, samples=3, sample_padding=0, magnitude_type="distance"): 
      super().__init__()
      ### in_channels + out_channels must be shuffle_scale**2
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.K = K
      self.stride = stride
      self.padding = padding
      self.shuffle_pattern = shuffle_pattern
      self.shuffle_scale = shuffle_scale
      self.samples = int(samples)
      self.sample_padding = sample_padding
      self.magnitude_type = magnitude_type
      
      self.upscale = PixelShuffle1D(upscale_factor=self.shuffle_scale)
      
      self.downscale = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
      
      self.Conv1d_NN_spatial = Conv1d_NN_spatial(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 K=self.K,
                                 stride=self.stride,
                                 padding=self.padding,
                                 shuffle_pattern=self.shuffle_pattern,
                                 shuffle_scale=self.shuffle_scale, 
                                 samples=self.samples, 
                                 magnitude_type=self.magnitude_type
                                 )
                                 
      
      
      self.flatten = nn.Flatten(start_dim=2)
      
      
   def forward(self, x): 
      # Ex. Original Size (32, 1, 28, 28) 
      x_ind = torch.round(torch.linspace(0 + self.sample_padding, x.shape[2] - self.sample_padding - 1, self.samples)).to(torch.int)
      y_ind = torch.round(torch.linspace(0 + self.sample_padding, x.shape[3] - self.sample_padding - 1, self.samples)).to(torch.int)
      x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
      x_sample = torch.flatten(x[:, :, x_grid, y_grid], 2) # shape [32, 1, 25] if sample == 5 
      
      # Flatten Layer : size (32, 1, 784)
      x1 = self.flatten(x)
      
      # Conv1d_NN Layer
      x2 = self.Conv1d_NN_spatial(x1, x_sample)
      
      # Unflatten Layer 
      unflatten = nn.Unflatten(dim=2, unflattened_size=x.shape[2:])
      x3 = unflatten(x2)

      return x3
   
   
# ex = torch.rand(32, 1, 28, 28) 
# print("Input: ", ex.shape)

# conv2d_nn_spatial = Conv2d_NN_spatial(in_channels=1, out_channels=3, K=3, stride=3, padding=0, shuffle_pattern="N/A", shuffle_scale=2, samples=5, sample_padding= 3, magnitude_type="similarity")
# output = conv2d_nn_spatial(ex)
# print("Output: ", output.shape) # [32, 3, 784]
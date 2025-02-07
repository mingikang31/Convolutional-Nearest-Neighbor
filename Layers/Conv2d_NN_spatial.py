'''Convolution 2D Spatial Nearest Neighbor Layer'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
from Conv1d_NN_spatial import Conv1d_NN_spatial
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D

import faiss
import numpy as np 



class Conv2d_NN_spatial(nn.Module): 
   def __init__(self, 
                in_channels, 
                out_channels,
                K=3, stride=3, 
                padding=0, 
                shuffle_pattern="BA", 
                shuffle_scale=2, 
                samples=3, 
                sample_padding=0, 
                magnitude_type="distance"): 
      
      
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
      
      if (self.shuffle_pattern in ["B", "BA"]):
         self.in_channels_1d = self.in_channels * (self.shuffle_scale**2)
         self.out_channels_1d = self.out_channels * (self.shuffle_scale**2)
      else:
         self.in_channels_1d = self.in_channels
         self.out_channels_1d = self.out_channels
      
      
      
      self.Conv1d_NN_spatial = Conv1d_NN_spatial(in_channels=self.in_channels_1d,
                                                   out_channels=self.out_channels_1d,
                                                   K=self.K,
                                                   stride=self.stride,
                                                   padding=self.padding,
                                                   shuffle_pattern="N/A",
                                                   shuffle_scale=1,
                                                   magnitude_type=self.magnitude_type
                                                   )
                                 
      
      
      self.flatten = nn.Flatten(start_dim=2)
      
      
   def forward(self, x): 
      
      if self.shuffle_pattern in ["B", "BA"]:
         x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
      else: 
         x1 = x
         
         
      # x sample matrix 
      x_ind = torch.round(torch.linspace(0 + self.sample_padding, x1.shape[2] - self.sample_padding - 1, self.samples)).to(torch.int)
      y_ind = torch.round(torch.linspace(0 + self.sample_padding, x1.shape[3] - self.sample_padding - 1, self.samples)).to(torch.int)
      
      x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
      
      x_idx_flat = x_grid.flatten()
      y_idx_flat = y_grid.flatten()
      
      width = x1.shape[2]
      
      # flat indices for indexing -> similar to random sampling for ConvNN
      flat_indices = x_idx_flat * width + y_idx_flat
      
      
      x_sample = torch.flatten(x1[:, :, x_grid, y_grid], 2)
      
      # input matrix
      x2 = self.flatten(x1)
      
      x3 = self.Conv1d_NN_spatial(x2, x_sample, flat_indices)
      
      unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
      x4 = unflatten(x3)
      
      if self.shuffle_pattern in ["A", "BA"]:
         x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
      else: 
         x5 = x4

      return x5
   
   
   
'''EXAMPLE USAGE'''
def example_usage():
   """Example Usage of Conv2d_NN_spatial Layer"""
   ex = torch.rand(32, 3, 28, 28) 
   print("Input: ", ex.shape)

   conv2d_nn_spatial = Conv2d_NN_spatial(in_channels=3, out_channels=3, K=3, stride=3, padding=0, shuffle_pattern="BA", shuffle_scale=2, samples=5, sample_padding= 3, magnitude_type="similarity")
   output = conv2d_nn_spatial(ex)
   print("Output: ", output.shape) # [32, 3, 784]
   
   

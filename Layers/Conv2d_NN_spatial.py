'''Convolution 2D Spatial Nearest Neighbor Layer'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
from Conv1d_NN_spatial import Conv1d_NN_spatial
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D

import faiss
import numpy as np 



class Conv2d_NN_spatial(nn.Module): 
   """
   - Location Channels : add coordinates -> unshuffle -> flatten -> ConvNN -> unflatten -> shuffle -> remove coordinate 
   """
   
   def __init__(self, 
                in_channels, 
                out_channels,
                K=3, stride=3, 
                padding=0, 
                shuffle_pattern="BA", 
                shuffle_scale=2, 
                samples=3, 
                sample_padding=0, 
                magnitude_type="similarity", 
                location_channels=False
                ): 
      
      
      super().__init__()
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
      
      
      self.Conv1d_NN_spatial = Conv1d_NN_spatial(in_channels=self.in_channels_1d,
                                                   out_channels=self.out_channels_1d,
                                                   K=self.K,
                                                   stride=self.stride,
                                                   padding=self.padding,
                                                   shuffle_pattern="NA",
                                                   shuffle_scale=1,
                                                   magnitude_type=self.magnitude_type
                                                   )
                                 
      
      self.flatten = nn.Flatten(start_dim=2)      
      
      
      self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)

      
      
   def forward(self, x): 
      
      if self.shuffle_pattern in ["B", "BA"]:
         if self.location_channels:
            x_coordinates = self.coordinate_channels(x.shape, device=x.device)
            x = torch.cat((x, x_coordinates), dim=1)
            x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
         else: 
            x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
         
      else: 
         if self.location_channels:
            x_coordinates = self.coordinate_channels(x.shape, device=x.device)
            x1 = torch.cat((x, x_coordinates), dim=1)
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
         if self.location_channels:
            x4 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
            x5 = self.pointwise_conv(x4)

         else:
            x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
      else: 
         if self.location_channels:
            x5 = self.pointwise_conv(x4)
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


class Conv2d_NN_spatial_prev(nn.Module): 
   """
    - Location Channels : unshuffle -> add coordinates -> flatten -> ConvNN -> unflatten -> remove coordinate -> shuffle
   """
   
   
   def __init__(self, 
                in_channels, 
                out_channels,
                K=3, stride=3, 
                padding=0, 
                shuffle_pattern="BA", 
                shuffle_scale=2, 
                samples=3, 
                sample_padding=0, 
                magnitude_type="similarity", 
                location_channels=False
                ): 
      
      
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
      self.location_channels = location_channels
      
      if (self.shuffle_pattern in ["B", "BA"]):
         self.in_channels_1d = self.in_channels * (self.shuffle_scale**2)
         self.out_channels_1d = self.out_channels * (self.shuffle_scale**2)
      else:
         self.in_channels_1d = self.in_channels
         self.out_channels_1d = self.out_channels
      
      if self.location_channels: 
         self.in_channels_1d += 2
         self.out_channels_1d += 2
      
      
      self.Conv1d_NN_spatial = Conv1d_NN_spatial(in_channels=self.in_channels_1d,
                                                   out_channels=self.out_channels_1d,
                                                   K=self.K,
                                                   stride=self.stride,
                                                   padding=self.padding,
                                                   shuffle_pattern="NA",
                                                   shuffle_scale=1,
                                                   magnitude_type=self.magnitude_type
                                                   )
                                 
      
      
      self.flatten = nn.Flatten(start_dim=2)      
      
      self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)

      
      
   def forward(self, x): 
      
      if self.shuffle_pattern in ["B", "BA"]:
         if self.location_channels:
            x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)          
            x1_coordinates = self.coordinate_channels(x1.shape, device=x.device)
            x1 = torch.cat((x1, x1_coordinates), dim=1)
            
         else: 
            x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)
         
      else: 
         if self.location_channels:
            x1_coordinates = self.coordinate_channels(x.shape, device=x.device)
            x1 = torch.cat((x, x1_coordinates), dim=1)
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
         if self.location_channels:
            x4 = self.pointwise_conv(x4)
            x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
         else:
            x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
      else: 
         if self.location_channels:
            x5 = self.pointwise_conv(x4)
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
   
   
   
'''EXAMPLE USAGE'''
def example_usage():
   """Example Usage of Conv2d_NN_spatial Layer"""
   ex = torch.rand(32, 3, 28, 28) 
   print("Input: ", ex.shape)

   conv2d_nn_spatial = Conv2d_NN_spatial(in_channels=3, out_channels=3, K=3, stride=3, padding=0, shuffle_pattern="BA", shuffle_scale=2, samples=5, sample_padding= 3, magnitude_type="similarity", location_channels=True)
   output = conv2d_nn_spatial(ex)
   print("Output: ", output.shape) # [32, 3, 784]
   
   a = conv2d_nn_spatial.coordinate_channels(ex.shape, device=ex.device)
   print("location_channels: ", a.shape)

# example_usage()
   
   
   

'''Convolution 2D Spatial Nearest Neighbor Layer'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
from Conv1d_NN_spatial import Conv1d_NN_spatial


class Conv2d_NN_spatial(nn.Module): 
   """
   - Location Channels : add coordinates -> unshuffle -> flatten -> ConvNN -> unflatten -> shuffle -> remove coordinate 
   """
   
   def __init__(self, 
                in_channels, 
                out_channels,
                K=3, 
                stride=3, 
                padding=0, 
                shuffle_pattern="BA", 
                shuffle_scale=2, 
                samples=3, 
                sample_padding=0, 
                magnitude_type="similarity", 
                location_channels=False
                ): 
      
      
      super(Conv2d_NN_spatial, self).__init__()
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
            
      self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
      self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
      
      self.Conv1d_NN_spatial = Conv1d_NN_spatial(in_channels=self.in_channels_1d,
                                                   out_channels=self.out_channels_1d,
                                                   K=self.K,
                                                   stride=self.stride,
                                                   padding=self.padding,
                                                   magnitude_type=self.magnitude_type
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
         
         
      # x sample matrix 
      x_ind = torch.round(torch.linspace(0 + self.sample_padding, x1.shape[2] - self.sample_padding - 1, self.samples)).to(torch.int)
      y_ind = torch.round(torch.linspace(0 + self.sample_padding, x1.shape[3] - self.sample_padding - 1, self.samples)).to(torch.int)
      
      x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
      
      x_idx_flat = x_grid.flatten()
      y_idx_flat = y_grid.flatten()      
            
      width = x1.shape[2]
      # flat indices for indexing -> similar to random sampling for ConvNN
      flat_indices = x_idx_flat * width + y_idx_flat
      
      x_sample = self.flatten(x1[:, :, x_grid, y_grid])
      
      # Input Matrix
      x2 = self.flatten(x1)
      
      x3 = self.Conv1d_NN_spatial(x2, x_sample, flat_indices.to(x.device))
      
      unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
      x4 = unflatten(x3)
      
      if self.shuffle_pattern in ["A", "BA"]:
         if self.location_channels:
            x4 = self.shuffle_layer(x4)
            x5 = self.pointwise_conv(x4)

         else:
            x5 = self.shuffle_layer(x4)
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

if __name__ == "__main__":
   x = torch.rand(32, 3, 32, 32)

   conv2d_nn_spatial = Conv2d_NN_spatial(in_channels=3, out_channels=8, K=3, stride=3, padding=0, samples=5, sample_padding= 3, magnitude_type="similarity", location_channels=True)
   output = conv2d_nn_spatial(x)
   
   print("Input shape:", x.shape) # Should be (32, 3, 32, 32)
   print("Output shape:", output.shape) # Should be (32, 8, 32, 32)
   
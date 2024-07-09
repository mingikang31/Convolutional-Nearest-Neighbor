'''Convolution 2D Nearest Neighbor Layer'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from Conv1d_NN import Conv1d_NN
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D



class Conv2d_NN(nn.Module): 
   def __init__(self, in_channels, out_channels, K=3, stride=3, padding=0, shuffle_pattern="N/A", shuffle_scale=2, neighbors="all"): 
      super().__init__()
      ### in_channels + out_channels must be shuffle_scale**2
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.K = K
      self.stride = stride
      self.padding = padding
      self.shuffle_pattern = shuffle_pattern
      self.shuffle_scale = shuffle_scale
      self.neighbors = neighbors
      
      
      self.upscale = PixelShuffle1D(upscale_factor=self.shuffle_scale)
      
      self.downscale = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
      
      self.Conv1d_NN = Conv1d_NN(in_channels=self.in_channels * shuffle_scale **2,
                                 out_channels=self.out_channels * shuffle_scale **2,
                                 K=self.K,
                                 stride=self.stride,
                                 padding=self.padding,
                                 shuffle_pattern=self.shuffle_pattern,
                                 shuffle_scale=self.shuffle_scale)
      
      # self.unshuffle = nn.functional.pixel_unshuffle(self.shuffle_scale)
      # self.shuffle = nn.functional.pixel_shuffle(self.shuffle_scale)
      
      self.flatten = nn.Flatten(start_dim=2)
      
      
   def forward(self, x): 
      # (32, 1, 28, 28) 
      
      # Unshuffle Layer 
      # Ex. (32, 16, 7, 7) if upscale_factor = 4
      x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)

      print("Unshuffle: ", x1.shape)
      
      # Flatten Layer 
      # Ex. (32, 16, 49) 
      x2 = self.flatten(x1)
      print("Flatten: ", x2.shape)
      
      # Conv1d_NN Layer
      # Ex. (32, 16, 49) 
      x3 = self.Conv1d_NN(x2)
      print("Conv1d_NN: ", x3.shape)
      
      # Unflatten Layer 
      # Ex. (32, 16, 7, 7)
      unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
      x4 = unflatten(x3)
      print("Unflatten: ", x4.shape)
      
      # Shuffle Layer 
      # Ex. (32, 16, 28, 28)
      x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
      print("Shuffle: ", x5.shape)
      return x5

'''EXAMPLE USAGE'''

# ex = torch.rand(32, 1, 28, 28) 
# print("Input: ", ex.shape)

# conv2d_nn = Conv2d_NN(in_channels=1, out_channels=3, K=3, stride=3, padding=0, shuffle_pattern="N/A", shuffle_scale=2, neighbors="all")
# output = conv2d_nn(ex)
# print("Output: ", output.shape)
      
      
   

      
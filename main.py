# Copyright (c) Mingi Kang | mkang817415. and its affiliates. All Rights Reserved
# Import or copy the code for testing. 

import torch 
from torch import nn 
from torch.nn import Conv1d, ReLU
import torch.nn.functional as F 
import faiss

import numpy as np 



'''
Conv2d_NN: Conv2d Nearest Neighbor Layer
Features:
    - For K nearest neighbors, K must be same as stride
    - Calculate Distance/Similarity Matrix for All Samples or N Samples
    - Pixel Shuffle/Unshuffle Layer with Scale Factor
    - Conv1d Layer with Kernel Size, Stride, Padding
'''
class Conv2d_NN(nn.Module): 
   def __init__(self, in_channels, out_channels, K=3, stride=3, padding=0, shuffle_pattern="N/A", shuffle_scale=2, samples="all", magnitude_type="distance"): 
      super().__init__()
      ### in_channels + out_channels must be shuffle_scale**2
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.K = K
      self.stride = stride
      self.padding = padding
      self.shuffle_pattern = shuffle_pattern
      self.shuffle_scale = shuffle_scale
      self.samples = samples
      self.magnitude_type = magnitude_type
      
      self.upscale = PixelShuffle1D(upscale_factor=self.shuffle_scale)
      
      self.downscale = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
      
      self.Conv1d_NN = Conv1d_NN(in_channels=self.in_channels * shuffle_scale **2,
                                 out_channels=self.out_channels * shuffle_scale **2,
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
      
      # Unshuffle Layer 
      # Ex. (32, 16, 7, 7) if upscale_factor = 4
      x1 = nn.functional.pixel_unshuffle(x, self.shuffle_scale)

      # print("Unshuffle: ", x1.shape)
      
      # Flatten Layer 
      # Ex. (32, 16, 49) 
      x2 = self.flatten(x1)
      # print("Flatten: ", x2.shape)
      
      # Conv1d_NN Layer
      # Ex. (32, 16, 49) 
      x3 = self.Conv1d_NN(x2)  
      # print("Conv1d_NN: ", x3.shape)
      
      # Unflatten Layer 
      # Ex. (32, 16, 7, 7)
      unflatten = nn.Unflatten(dim=2, unflattened_size=x1.shape[2:])
      x4 = unflatten(x3)
      # print("Unflatten: ", x4.shape)
      
      # Shuffle Layer 
      # Ex. (32, 16, 28, 28)
      x5 = nn.functional.pixel_shuffle(x4, self.shuffle_scale)
      # print("Shuffle: ", x5.shape)
      return x5

'''
Conv2d_NN_spatial: Conv2d Nearest Neighbor Spatial Layer
Features: 
    - For K nearest neighbors, K must be same as stride
    - Calculate Distance/Similarity Matrix with matrix and matrix2 
    - Pixel Shuffle/Unshuffle Layer with Scale Factor
    - Conv1d Layer with Kernel Size, Stride, Padding
'''
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
   
'''
Conv1d_NN: Conv1d Nearest Neighbor Layer
Features: 
    - For K nearest neighbors, K must be same as stride
    - Calculate Distance/Similarity Matrix for All Samples or N Samples
    - Pixel Shuffle/Unshuffle Layer with Scale Factor
    - Conv1d Layer with Kernel Size, Stride, Padding 
'''
class Conv1d_NN(nn.Module): 
    def __init__(self, in_channels, out_channels, K=3, stride=3, padding=0, 
                 shuffle_pattern='N/A', shuffle_scale=2, 
                 samples='all', 
                 magnitude_type='distance'): 
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride 
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern 
        self.shuffle_scale = shuffle_scale
        
        self.samples = int(samples) if samples != 'all' else samples # Number of samples to consider
        self.magnitude_type = magnitude_type # Nearest Neighbor based on Distance or Similarity 
        self.maximum = True if self.magnitude_type == 'similarity' else False # Minimum or Maximum for Distance or Similarity
        
        # Unshuffle layer 
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        
        # Shuffle Layer 
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
                
        # Conv1d Layer
        self.in_channels = in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels

        # Conv1d Layer 
        self.conv1d_layer = Conv1d(in_channels=self.in_channels, 
                                    out_channels=self.out_channels, 
                                    kernel_size=self.K, 
                                    stride=self.stride, 
                                    padding=self.padding)

        self.relu = ReLU()

    def forward(self, x): 
        # Consider all samples 
        if self.samples == 'all': 
            # Unshuffle Layer 
            if self.shuffle_pattern in ["B", "BA"]:
                x1 = self.unshuffle_layer(x)
            else:
                x1 = x
            
            # Calculate Distance/Similarity Matrix + Prime Vmap 2D
            if self.magnitude_type == 'distance': 
                matrix_magnitude = self.calculate_distance_matrix(x1)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self.calculate_similarity_matrix(x1)
                
            prime_2d = self.prime_vmap_2d(x1, matrix_magnitude, self.K, self.maximum) 
            
            # Conv1d Layer
            x2 = self.conv1d_layer(prime_2d)
            
            # ReLU Activation
            x3 = self.relu(x2)
            
            # Shuffle Layer 
            if self.shuffle_pattern in ["A", "BA"]:
                x4 = self.shuffle_layer(x3)
            else:
                x4 = x3
            
            return x4
        
        # Consider N samples
        else: 
            # Unshuffle Layer 
            if self.shuffle_pattern in ["B", "BA"]:
                x1 = self.unshuffle_layer(x)
            else:
                x1 = x
                
            # Calculate Distance/Similarity Matrix + Prime Vmap 2D
            rand_idx = torch.tensor(np.random.choice(x1.shape[2], self.samples, replace=False)) # list 
            x1_sample = x1[:, :, rand_idx]
            
            if self.magnitude_type == 'distance':
                matrix_magnitude = self.calculate_distance_matrix_N(x1, x1_sample)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self.calculate_similarity_matrix_N(x1, x1_sample)
                
            if self.magnitude_type == 'distance':
                matrix_magnitude[:, rand_idx, np.arange(len(rand_idx))] = np.inf 
            elif self.magnitude_type == 'similarity':
                matrix_magnitude[:, rand_idx, np.arange(len(rand_idx))] = -np.inf
                
            
            prime = self.prime_vmap_2d_N(x1, matrix_magnitude, self.K, rand_idx, self.maximum)
            
            # Conv1d Layer
            x2 = self.conv1d_layer(prime)
            
            # ReLU Activation
            x3 = self.relu(x2)
            
            # Shuffle Layer
            if self.shuffle_pattern in ["A", "BA"]:
                x4 = self.shuffle_layer(x3)
            else:
                x4 = x3
            
            return x4
    
    ### All Neighbors ###
    '''Distance Matrix Calculations for All Sample'''
    @staticmethod
    def calculate_distance_matrix(matrix):
        '''Calculating distance matrix of the input matrix'''
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix)
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
        return torch.sqrt(dist_matrix)

    '''Similarity Matrix Calculations for All Sample'''
    @staticmethod 
    def calculate_similarity_matrix(matrix): 
        '''Calculate the similarity matrix of the input matrix'''
        normalized_matrix = F.normalize(matrix, p=2, dim=1) # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        dot_product = torch.bmm(normalized_matrix.transpose(2, 1), normalized_matrix)
        similarity_matrix = dot_product 
        return similarity_matrix
    
    '''All Sample Methods'''
    @staticmethod 
    def prime_vmap_2d(matrix, magnitude_matrix, num_nearest_neighbors, maximum): 
        # Vectorization / Vmap Implementation for Nearest Neighbor Tensor 2D 
        batched_process = torch.vmap(Conv1d_NN.process_batch, in_dims=(0, 0, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, flatten=True, maximum=maximum)
        return prime 

    @staticmethod 
    def prime_vmap_3d(matrix, magnitude_matrix, num_nearest_neighbors, maximum): 
        # Vectorization / Vmap Implementation for Nearest Neighbor Tensor 3D
        batched_process = torch.vmap(Conv1d_NN.process_batch, in_dims=(0, 0, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, flatten=False, maximum=maximum)
        return prime

    @staticmethod 
    def process_batch(matrix, magnitude_matrix, num_nearest_neighbors, flatten, maximum): 
        # Process the batch of matrices
        ind = torch.topk(magnitude_matrix, num_nearest_neighbors, largest=maximum).indices 
        neigh = matrix[:, ind]
        if flatten: 
            reshape = torch.flatten(neigh, start_dim=1)
            return reshape
        return neigh
    
    ### N Samples ### 
    '''Distance Matrix Calculations for N Sample'''
    @staticmethod 
    def calculate_distance_matrix_N(matrix, matrix_sample):
        '''Calculate distance matrix between two input matrices''' 
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_sample = torch.sum(matrix_sample ** 2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix_sample)
        dist_matrix = norm_squared + norm_squared_sample - 2 * dot_product
        return torch.sqrt(dist_matrix)
        
    '''Similarity Matrix Calculations for N Sample'''
    @staticmethod
    def calculate_similarity_matrix_N(matrix, matrix_sample): 
        normalized_matrix = F.normalize(matrix, p=2, dim=1) # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        normalized_matrix_sample = F.normalize(matrix_sample, p=2, dim=1)
        similarity_matrix = dot_product = torch.bmm(normalized_matrix.transpose(2, 1), normalized_matrix_sample)
        return similarity_matrix

    '''N Sample Methods'''
    @staticmethod
    def prime_vmap_2d_N(matrix, magnitude_matrix, num_nearest_neighbors, rand_idx, maximum): 
        '''Vectorization / Vmap Implementation for Nearest Neighbor Tensor 2D'''
        batched_process = torch.vmap(Conv1d_NN.process_batch_N, in_dims=(0, 0, None, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, rand_idx, flatten=True, maximum=maximum)
        return prime 
    
    @staticmethod
    def prime_vmap_3d_N(matrix, magnitude_matrix, num_nearest_neighbors, rand_idx, maximum): 
        '''Vectorization / Vmap Implementation for Nearest Neighbor Tensor 3D'''
        batched_process = torch.vmap(Conv1d_NN.process_batch_N, in_dims=(0, 0, None, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, rand_idx, flatten=False, maximum=maximum)
        return prime
    
    @staticmethod
    def process_batch_N(matrix, magnitude_matrix, num_nearest_neighbors, rand_idx, flatten, maximum): 
        '''Process the batch of matrices'''
        topk_ind = torch.topk(magnitude_matrix, num_nearest_neighbors - 1, largest=maximum).indices
        device = topk_ind.device
        rand_idx = rand_idx.to(device) # same device as topk_ind
        mapped_tensor = rand_idx[topk_ind] 
        index_tensor = torch.arange(0, matrix.shape[1], device=device).unsqueeze(1) # shape [40, 1]
        final_tensor = torch.cat([index_tensor, mapped_tensor], dim=1)
        neigh = matrix[:, final_tensor] 
        if flatten: 
            reshape = torch.flatten(neigh, start_dim=1)
            return reshape
        return neigh
    
'''
Conv1d_NN_spatial: Conv1d Nearest Neighbor Spatial Layer
Features: 
    - For K nearest neighbors, K must be same as stride
    - Calculate Distance/Similarity Matrix with matrix and matrix2 
    - Pixel Shuffle/Unshuffle Layer with Scale Factor
    - Conv1d Layer with Kernel Size, Stride, Padding
'''
class Conv1d_NN_spatial(nn.Module): 
    def __init__(self, in_channels, out_channels, K=3, stride=3, padding=0, 
                 shuffle_pattern='N/A', shuffle_scale=2, 
                 samples='all', 
                 magnitude_type='distance'): 
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride 
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern 
        self.shuffle_scale = shuffle_scale
        
        self.samples = int(samples) if samples != 'all' else samples # Number of samples to consider
        self.magnitude_type = magnitude_type # Nearest Neighbor based on Distance or Similarity 
        self.maximum = True if self.magnitude_type == 'similarity' else False # Minimum or Maximum for Distance or Similarity
        
        # Unshuffle layer 
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        
        # Shuffle Layer 
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
                
        # Conv1d Layer
        self.in_channels = in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels

        # Conv1d Layer 
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.K, 
                                      stride=self.stride, 
                                      padding=self.padding)

        self.relu = nn.ReLU()

    def forward(self, x, y): 

        # Unshuffle Layer 
        if self.shuffle_pattern in ["B", "BA"]:
            x1 = self.unshuffle_layer(x)
        else:
            x1 = x
            
        if self.magnitude_type == 'distance':
            matrix_magnitude = self.calculate_distance_matrix_N(x1, y)
        elif self.magnitude_type == 'similarity':
            matrix_magnitude = self.calculate_similarity_matrix_N(x1, y)        
        
        prime = self.prime_vmap_2d_N(x1, matrix_magnitude, self.K, self.maximum)
        
        # Conv1d Layer
        x2 = self.conv1d_layer(prime)
        
        # ReLU Activation
        x3 = self.relu(x2)
        
        # Shuffle Layer
        if self.shuffle_pattern in ["A", "BA"]:
            x4 = self.shuffle_layer(x3)
        else:
            x4 = x3
        
        return x4
        
    ### N Samples ### 
    '''Distance Matrix Calculations for N Sample'''
    @staticmethod 
    def calculate_distance_matrix_N(matrix, matrix_sample):
        '''Calculate distance matrix between two input matrices''' 
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_sample = torch.sum(matrix_sample ** 2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix_sample)
        dist_matrix = norm_squared + norm_squared_sample - 2 * dot_product
        return torch.sqrt(dist_matrix)
        
    '''Similarity Matrix Calculations for N Sample'''
    @staticmethod
    def calculate_similarity_matrix_N(matrix, matrix_sample): 
        normalized_matrix = F.normalize(matrix, p=2, dim=1) # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        normalized_matrix_sample = F.normalize(matrix_sample, p=2, dim=1)
        similarity_matrix = dot_product = torch.bmm(normalized_matrix.transpose(2, 1), normalized_matrix_sample)
        return similarity_matrix

    '''N Sample Methods'''
    @staticmethod
    def prime_vmap_2d_N(matrix, magnitude_matrix, num_nearest_neighbors, maximum): 
        '''Vectorization / Vmap Implementation for Nearest Neighbor Tensor 2D'''
        batched_process = torch.vmap(Conv1d_NN_spatial.process_batch_N, in_dims=(0, 0, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, flatten=True, maximum=maximum)
        return prime 
    
    @staticmethod
    def prime_vmap_3d_N(matrix, magnitude_matrix, num_nearest_neighbors, maximum): 
        '''Vectorization / Vmap Implementation for Nearest Neighbor Tensor 3D'''
        batched_process = torch.vmap(Conv1d_NN_spatial.process_batch_N, in_dims=(0, 0, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, flatten=False, maximum=maximum)
        return prime
    
    @staticmethod 
    def process_batch_N(matrix, magnitude_matrix, num_nearest_neighbors, flatten, maximum): 
        # Process the batch of matrices
        ind = torch.topk(magnitude_matrix, num_nearest_neighbors, largest=maximum).indices 
        neigh = matrix[:, ind]
        if flatten: 
            reshape = torch.flatten(neigh, start_dim=1)
            return reshape
        return neigh
    
#####Pixel Shuffle/Unshuffle 1D####################################
class PixelShuffle1D(nn.Module): 
   def __init__(self, upscale_factor):
      super(PixelShuffle1D, self).__init__()
      
      # input's channel must be divisible by the upscale factor
      self.upscale_factor = upscale_factor
   
   def forward(self, x): 
      batch_size, channel_len, token_len = x.shape[0], x.shape[1], x.shape[2]
      
      output_channel_len = channel_len / self.upscale_factor 
      if output_channel_len.is_integer() == False: 
         raise ValueError('Input channel length must be divisible by upscale factor')
      output_channel_len = int(output_channel_len)
      
      output_token_len = int(token_len * self.upscale_factor)
      
      x = torch.reshape(x, (batch_size, output_channel_len, output_token_len)).contiguous()
      
      return x 
   
class PixelUnshuffle1D(nn.Module):  
   def __init__(self, downscale_factor):
      super(PixelUnshuffle1D, self).__init__()
      
      self.downscale_factor = downscale_factor

   def forward(self, x):
      batch_size = x.shape[0]
      channel_len = x.shape[1]
      token_len = x.shape[2]

      output_channel_len = int(channel_len * self.downscale_factor)
      output_token_len = token_len / self.downscale_factor
      
      if output_token_len.is_integer() == False:
         raise ValueError('Input token length must be divisible by downscale factor')
      output_token_len = int(output_token_len)
      
      x = torch.reshape(x, (batch_size, output_channel_len, output_token_len)).contiguous()
      
      return x 


import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F 

from functools import partial

from timm.models.vision_transformer import _cfg, VisionTransformer 
from timm.models.registry import register_model

from torchsummary import summary

'''Layer Modules for ConvNN, ConvNN_Spatial, ConvNN_Attn, PixelShuffle1D, PixelUnshuffle1D'''
### Convolutional Nearest Neighbors ###

class Conv1d_NN(nn.Module): 
    """
    Convolution 1D Nearest Neighbor Layer
    
    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        K (int): Number of Nearest Neighbors for consideration.
        stride (int): Stride size.
        padding (int): Padding size.
        shuffle_pattern (str): Shuffle pattern.
        shuffle_scale (int): Shuffle scale factor.
        samples (int/str): Number of samples to consider.
        magnitude_type (str): Distance or Similarity.
        
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 K=3, 
                 stride=3, 
                 padding=0, 
                 shuffle_pattern='N/A', 
                 shuffle_scale=2, 
                 samples='all', 
                 magnitude_type='similarity'
                 ): 
        
        """
        Initializes the Conv1d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
        
        super(Conv1d_NN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride 
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern 
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != 'all' else samples 
        self.magnitude_type = magnitude_type 
        self.maximum = True if self.magnitude_type == 'similarity' else False
        
        # Unshuffle layer 
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        
        # Shuffle Layer 
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        
        # Channels for Conv1d Layer
        self.in_channels = in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels

        # Conv1d Layer 
        self.conv1d_layer = Conv1d(in_channels=self.in_channels, 
                                    out_channels=self.out_channels, 
                                    kernel_size=self.K, 
                                    stride=self.stride, 
                                    padding=self.padding)

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
                matrix_magnitude = self._calculate_distance_matrix(x1)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix(x1)
                
            prime_2d = self._prime(x1, matrix_magnitude, self.K, self.maximum) 
            
            # Conv1d Layer
            x2 = self.conv1d_layer(prime_2d)
            
            
            # Shuffle Layer 
            if self.shuffle_pattern in ["A", "BA"]:
                x3 = self.shuffle_layer(x2)
            else:
                x3 = x2
            
            return x3
        
        # Consider N samples
        else: 
            # Unshuffle Layer 
            if self.shuffle_pattern in ["B", "BA"]:
                x1 = self.unshuffle_layer(x)
            else:
                x1 = x
                
            # Calculate Distance/Similarity Matrix + Prime       
            rand_idx = torch.randperm(x1.shape[2], device=x1.device)[:self.samples]
            
            x1_sample = x1[:, :, rand_idx]
            
            if self.magnitude_type == 'distance':
                matrix_magnitude = self._calculate_distance_matrix_N(x1, x1_sample)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix_N(x1, x1_sample)
                
            range_idx = torch.arange(len(rand_idx), device=x1.device)
                
        
            if self.magnitude_type == 'distance':
                matrix_magnitude[:, rand_idx, range_idx] = float('inf') 
            elif self.magnitude_type == 'similarity':
                matrix_magnitude[:, rand_idx, range_idx] = float('-inf')
                
            
            prime = self._prime_N(x1, matrix_magnitude, self.K, rand_idx, self.maximum)
            
            # Conv1d Layer
            x2 = self.conv1d_layer(prime)
            
            # Shuffle Layer
            if self.shuffle_pattern in ["A", "BA"]:
                x3 = self.shuffle_layer(x2)
            else:
                x3 = x2
            
            return x3
    
    
    def _calculate_distance_matrix(self, matrix, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix)
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0) # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        return dist_matrix
    
    def _calculate_distance_matrix_N(self, matrix, matrix_sample, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_sample = torch.sum(matrix_sample ** 2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix_sample)
        
        dist_matrix = norm_squared + norm_squared_sample - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0) # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        return dist_matrix
    
    
    def _calculate_similarity_matrix(self, matrix):
        norm_matrix = F.normalize(matrix, p=2, dim=1) # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        similarity_matrix = torch.bmm(norm_matrix.transpose(2, 1), norm_matrix)
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, matrix, matrix_sample):
        norm_matrix = F.normalize(matrix, p=2, dim=1) # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        norm_sample = F.normalize(matrix_sample, p=2, dim=1)
        similarity_matrix = torch.bmm(norm_matrix.transpose(2, 1), norm_sample)
        return similarity_matrix

    def _prime(self, matrix, magnitude_matrix, K, maximum):
        b, c, t = matrix.shape
        # Get top-K indices: shape [b, t, K]
        _, topk_indices = torch.topk(magnitude_matrix, k=K, dim=2, largest=maximum)
        
        # Expand indices to add channel dimension: [b, 1, t, K] then expand to [b, c, t, K]
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        
        # Unsqueeze matrix and expand so that the gathered dimension has size K.
        # matrix.unsqueeze(-1) yields shape [b, c, t, 1]
        # Then expand to [b, c, t, K] and force contiguous memory.
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        
        # Gather along the token dimension (dim=2) using the expanded indices.
        prime = torch.gather(matrix_expanded, dim=2, index=topk_indices_exp)
        
        # Flatten the token and neighbor dimensions: [b, c, t*K]
        prime = prime.view(b, c, -1)
        return prime
    
    def _prime_N(self, matrix, magnitude_matrix, K, rand_idx, maximum):
        b, c, t = matrix.shape

        # Get top-(K-1) indices from the magnitude matrix; shape: [b, t, K-1]
        _, topk_indices = torch.topk(magnitude_matrix, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # Map indices from the sampled space to the full token indices using rand_idx.
        # mapped_tensor will have shape: [b, t, K-1]
        mapped_tensor = rand_idx[topk_indices]

        # Create self indices for each token; shape: [1, t, 1] then expand to [b, t, 1]
        token_indices = torch.arange(t, device=matrix.device).view(1, t, 1).expand(b, t, 1)

        # Concatenate self index with neighbor indices to form final indices; shape: [b, t, K]
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)

        # Expand final_indices to include the channel dimension; result shape: [b, c, t, K]
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Expand matrix to shape [b, c, t, 1] and then to [b, c, t, K] (ensuring contiguous memory)
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()

        # Gather neighbor features along the token dimension (dim=2)
        prime = torch.gather(matrix_expanded, dim=2, index=indices_expanded)  # shape: [b, c, t, K]

        # Flatten the token and neighbor dimensions into one: [b, c, t*K]
        prime = prime.view(b, c, -1)
        return prime

class Conv2d_NN(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer for Convolutional Neural Networks.
     - Location Channels : add coordinates -> unshuffle -> flatten -> ConvNN -> unflatten -> shuffle -> remove coordinate 
    
    Attributes: 
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        K (int): Number of Nearest Neighbors for consideration.
        stride (int): Stride size.
        padding (int): Padding size.
        shuffle_pattern (str): Shuffle pattern.
        shuffle_scale (int): Shuffle scale factor.
        samples (int/str): Number of samples to consider.
        magnitude_type (str): Distance or Similarity.
        
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K=3,
                stride=3, 
                padding=0, 
                shuffle_pattern="BA", 
                shuffle_scale=2, 
                samples="all", 
                magnitude_type="similarity",
                location_channels=False
                ): 
        
        """
        Initializes the Conv2d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
    
        super(Conv2d_NN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != "all" else samples
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


        self.Conv1d_NN = Conv1d_NN(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples, 
                                    shuffle_pattern="NA",
                                    shuffle_scale=1, 
                                    magnitude_type=self.magnitude_type
                                    )

        self.flatten = nn.Flatten(start_dim=2)
        
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        

        self.pointwise_conv = nn.Conv2d(self.out_channels + 2, self.out_channels, kernel_size=1)
        
        self.coord_cache = {}
        
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

        x3 = self.Conv1d_NN(x2)  

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
        cache_key = f"{tensor_shape[2]}_{tensor_shape[3]}_{device}"
        if cache_key in self.coord_cache:
            return self.coord_cache[cache_key]
        
        
        x_ind = torch.arange(0, tensor_shape[2])
        y_ind = torch.arange(0, tensor_shape[3])
        
        x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
        x_grid = x_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        y_grid = y_grid.float().unsqueeze(0).expand(tensor_shape[0], -1, -1).unsqueeze(1)
        
        xy_grid = torch.cat((x_grid, y_grid), dim=1)
        xy_grid_normalized = F.normalize(xy_grid, p=2, dim=1)
        self.coord_cache[cache_key] = xy_grid_normalized.to(device)
        
        return xy_grid_normalized.to(device)    

class Conv1d_NN_spatial(nn.Module): 
    """
    Convolutional 1D Spatially Located Nearest Neighbor Layer for Convolutional Neural Networks.
    
    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        K (int): Number of Nearest Neighbors
        stride (int): Stride size.
        padding (int): Padding size.
        shuffle_pattern (str): Shuffle pattern.
        shuffle_scale (int): Shuffle scale factor.
        samples (int). Number of samples to consider.
        magnitude_type (str): Distance or Similarity.
    
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 K, 
                 stride, 
                 padding, 
                 magnitude_type='similarity'
                 ): 
        
        """
        Initializes the Conv1d_NN_spatial module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            magnitude_type (str): Distance or Similarity.
        """
        
        super(Conv1d_NN_spatial, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride 
        self.padding = padding        

        self.magnitude_type = magnitude_type 
        self.maximum = True if self.magnitude_type == 'similarity' else False 
    
        # Conv1d Layer 
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.K, 
                                      stride=self.stride, 
                                      padding=self.padding)

    def forward(self, x, y, indices): 
            
        if self.magnitude_type == 'distance':
            matrix_magnitude = self.calculate_distance_matrix_N(x, y)
        elif self.magnitude_type == 'similarity':
            matrix_magnitude = self.calculate_similarity_matrix_N(x, y)        
        
        prime = self.prime_vmap_2d_N(x, matrix_magnitude, self.K, indices, self.maximum)
        
        # Conv1d Layer
        x2 = self.conv1d_layer(prime)
        
        return x2
    
    
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
    def prime_vmap_2d_N(matrix, magnitude_matrix, num_nearest_neighbors, spatial_idx, maximum): 
        """Vectorization / Vmap Implementation for Nearest Neighbor Tensor 2D"""
        batched_process = torch.vmap(Conv1d_NN_spatial.process_batch_N, in_dims=(0, 0, None, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, spatial_idx, flatten=True, maximum=maximum)
        return prime 
    
    @staticmethod
    def prime_vmap_3d_N(matrix, magnitude_matrix, num_nearest_neighbors, spatial_idx, maximum): 
        """Vectorization / Vmap Implementation for Nearest Neighbor Tensor 3D"""
        batched_process = torch.vmap(Conv1d_NN_spatial.process_batch_N, in_dims=(0, 0, None, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, spatial_idx, flatten=False, maximum=maximum)
        return prime
    
    @staticmethod
    def process_batch_N(matrix, magnitude_matrix, num_nearest_neighbors, spatial_idx, flatten, maximum): 
        """Process the batch of matrices by finding the K nearest neighbors with reshaping."""
        topk_ind = torch.topk(magnitude_matrix, num_nearest_neighbors - 1, largest=maximum).indices
        device = topk_ind.device
        spatial_idx = spatial_idx.to(device) # same device as topk_ind
        mapped_tensor = spatial_idx[topk_ind] 
        index_tensor = torch.arange(0, matrix.shape[1], device=device).unsqueeze(1) # shape [40, 1]
        final_tensor = torch.cat([index_tensor, mapped_tensor], dim=1)
        neigh = matrix[:, final_tensor] 
        if flatten: 
            reshape = torch.flatten(neigh, start_dim=1)
            return reshape
        return neigh
    
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

class Conv1d_NN_Attn(nn.Module):
    """
    Convolutional 1D Nearest Neighbors Attention Layer 
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K=3, 
                stride=3, 
                padding=0, 
                shuffle_pattern='N/A', 
                shuffle_scale=1, 
                samples='all', 
                magnitude_type='similarity', 
                num_tokens = 224
                ): 
        
        """
        Initializes the Conv1d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
        
        super(Conv1d_NN_Attn, self).__init__()
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride 
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern 
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != 'all' else samples 
        self.magnitude_type = magnitude_type 
        self.maximum = True if self.magnitude_type == 'similarity' else False
        self.num_tokens =  int(num_tokens / 2) if self.shuffle_pattern in ["B", "BA"] else num_tokens        
        
        # Unshuffle layer 
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        
        # Shuffle Layer 
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        
        # Channels for Conv1d Layer
        self.in_channels = in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels

        # Conv1d Layer 
        self.conv1d_layer = Conv1d(in_channels=self.in_channels, 
                                    out_channels=self.out_channels, 
                                    kernel_size=self.K, 
                                    stride=self.stride, 
                                    padding=self.padding)
        
        # Linear Layer for Query, Key, Value
        self.w_q = nn.Linear(self.num_tokens, self.num_tokens, bias=False)
        self.w_k = nn.Linear(self.num_tokens, self.num_tokens, bias=False)
        self.w_v = nn.Linear(self.num_tokens, self.num_tokens, bias=False)

        
        
        
    def forward(self, x): 
        # Consider all samples 
        if self.samples == 'all': 
            # Unshuffle Layer 
            if self.shuffle_pattern in ["B", "BA"]:
                x1 = self.unshuffle_layer(x)
            else:
                x1 = x
            
            # Q, K, V 
            q = self.w_q(x1)
            k = self.w_k(x1)
            v = self.w_v(x1)
            
            # Calculate Distance/Similarity Matrix + Prime Vmap 2D
            if self.magnitude_type == 'distance': 
                matrix_magnitude = self._calculate_distance_matrix(k, q, sqrt=True)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix(k, q)
                
            prime_2d = self._prime(v, matrix_magnitude, self.K, self.maximum) 
            
            # Conv1d Layer
            x2 = self.conv1d_layer(prime_2d)
            
            
            # Shuffle Layer 
            if self.shuffle_pattern in ["A", "BA"]:
                x3 = self.shuffle_layer(x2)
            else:
                x3 = x2
            
            return x3
        
        # Consider N samples
        else: 
            # Unshuffle Layer 
            if self.shuffle_pattern in ["B", "BA"]:
                x1 = self.unshuffle_layer(x)
            else:
                x1 = x
                
            # Q, K, V 
            q = self.w_q(x1)
            k = self.w_k(x1)
            v = self.w_v(x1)
            
                
            # Calculate Distance/Similarity Matrix + Prime       
            rand_idx = torch.randperm(x1.shape[2], device=x1.device)[:self.samples]
            
            q_sample = q[:, :, rand_idx]
            
            if self.magnitude_type == 'distance':
                matrix_magnitude = self._calculate_distance_matrix_N(k, q_sample, sqrt=True)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix_N(k, q_sample)
                
            range_idx = torch.arange(len(rand_idx), device=x1.device)
                
        
            if self.magnitude_type == 'distance':
                matrix_magnitude[:, rand_idx, range_idx] = float('inf') 
            elif self.magnitude_type == 'similarity':
                matrix_magnitude[:, rand_idx, range_idx] = float('-inf')
                
            
            prime = self._prime_N(v, matrix_magnitude, self.K, rand_idx, self.maximum)
            
            # Conv1d Layer
            x2 = self.conv1d_layer(prime)
            
            # Shuffle Layer
            if self.shuffle_pattern in ["A", "BA"]:
                x3 = self.shuffle_layer(x2)
            else:
                x3 = x2
            
            return x3
        

    def _calculate_similarity_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  # [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  # remove negative values
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  # [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  # remove negative values
        return similarity_matrix
        

    def _calculate_distance_matrix(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True) 
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True) 
        
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        
        # Broadcasting: [B, 1, N] + [B, M, 1] - 2*[B, N, M]
        dist_matrix = norm_squared_K + norm_squared_Q.transpose(2, 1) - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        
        return dist_matrix

    def _calculate_distance_matrix_N(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        
        # Broadcasting: [B, 1, N] + [B, M, 1] - 2*[B, N, M]
        dist_matrix = norm_squared_K + norm_squared_Q - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        
        return dist_matrix

    def _prime(self, v, qk, K, maximum):
        b, c, t = v.shape 
        
        _, topk_indices = torch.topk(qk, k=K, dim=-1, largest = maximum)
        
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K)
        
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        
        prime = prime.reshape(b, c, -1)

        return prime
        
            
            
    def _prime_N(self, v, qk, K, rand_idx, maximum):
        b, c, t = v.shape

        # Get top-(K-1) indices from the magnitude matrix; shape: [b, t, K-1]
        _, topk_indices = torch.topk(qk, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # Map indices from the sampled space to the full token indices using rand_idx.
        # mapped_tensor will have shape: [b, t, K-1]
        mapped_tensor = rand_idx[topk_indices]

        # Create self indices for each token; shape: [1, t, 1] then expand to [b, t, 1]
        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)

        # Concatenate self index with neighbor indices to form final indices; shape: [b, t, K]
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)

        # Expand final_indices to include the channel dimension; result shape: [b, c, t, K]
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Expand matrix to shape [b, c, t, 1] and then to [b, c, t, K] (ensuring contiguous memory)
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()

        # Gather neighbor features along the token dimension (dim=2)
        prime = torch.gather(v_expanded, dim=2, index=indices_expanded)  # shape: [b, c, t, K]

        # Flatten the token and neighbor dimensions into one: [b, c, t*K]
        prime = prime.reshape(b, c, -1)
        return prime

class Conv2d_NN_Attn(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer for Convolutional Neural Networks.
     - Location Channels : add coordinates -> unshuffle -> flatten -> ConvNN -> unflatten -> shuffle -> remove coordinate 
    
    Attributes: 
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        K (int): Number of Nearest Neighbors for consideration.
        stride (int): Stride size.
        padding (int): Padding size.
        shuffle_pattern (str): Shuffle pattern.
        shuffle_scale (int): Shuffle scale factor.
        samples (int/str): Number of samples to consider.
        magnitude_type (str): Distance or Similarity.
        
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K=3,
                stride=3, 
                padding=0, 
                shuffle_pattern="BA", 
                shuffle_scale=2, 
                samples="all", 
                magnitude_type="similarity",
                location_channels=False, 
                image_size=(32, 32)
                ): 
        
        """
        Initializes the Conv2d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
        
        super(Conv2d_NN_Attn, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != "all" else samples
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



        self.num_tokens = int((image_size[0] * image_size[1]) / (self.shuffle_scale**2))

        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        
        self.Conv1d_NN = Conv1d_NN_Attn(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples, 
                                    shuffle_pattern="NA",
                                    shuffle_scale=1, 
                                    magnitude_type=self.magnitude_type, 
                                    num_tokens=self.num_tokens
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

        x3 = self.Conv1d_NN(x2)  

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

class Conv1d_NN_Attn_spatial(nn.Module):
    """
    Convolutional 1D Nearest Neighbors Attention Layer 
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K, 
                stride, 
                padding, 
                samples, 
                magnitude_type='similarity', 
                num_tokens = 224
                ): 
        
        """
        Initializes the Conv1d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
        
        super(Conv1d_NN_Attn_spatial, self).__init__()
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride 
        self.padding = padding
        self.samples = int(samples) 
        self.magnitude_type = magnitude_type 
        self.maximum = True if self.magnitude_type == 'similarity' else False
        
        self.num_tokens = num_tokens        
        
        # Conv1d Layer 
        self.conv1d_layer = Conv1d(in_channels=self.in_channels, 
                                    out_channels=self.out_channels, 
                                    kernel_size=self.K, 
                                    stride=self.stride, 
                                    padding=self.padding)
        
        # Linear Layer for Query, Key, Value
        self.w_q = nn.Linear(self.samples, self.samples, bias=False)
        self.w_k = nn.Linear(self.num_tokens, self.num_tokens, bias=False)
        self.w_v = nn.Linear(self.num_tokens, self.num_tokens, bias=False)

    def forward(self, x, y, indices): 
        # Q, K, V 
        q = self.w_q(y)
        k = self.w_k(x)
        v = self.w_v(x)
        
        if self.magnitude_type == 'distance':
            matrix_magnitude = self._calculate_distance_matrix_N(k, q, sqrt=True)
        elif self.magnitude_type == 'similarity':
            matrix_magnitude = self._calculate_similarity_matrix_N(k, q)
                    
        prime = self._prime_N(v, matrix_magnitude, self.K, indices, self.maximum)
                
        # Conv1d Layer
        x2 = self.conv1d_layer(prime)
        
        return x2
        

    def _calculate_similarity_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  # [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  # remove negative values
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  # [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  # remove negative values
        return similarity_matrix
        

    def _calculate_distance_matrix(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True) 
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True) 
        
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        
        # Broadcasting: [B, 1, N] + [B, M, 1] - 2*[B, N, M]
        dist_matrix = norm_squared_K + norm_squared_Q.transpose(2, 1) - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        
        return dist_matrix

    def _calculate_distance_matrix_N(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        
        # Broadcasting: [B, 1, N] + [B, M, 1] - 2*[B, N, M]
        dist_matrix = norm_squared_K + norm_squared_Q - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        
        return dist_matrix

    def _prime(self, v, qk, K, maximum):
        b, c, t = v.shape 
        
        _, topk_indices = torch.topk(qk, k=K, dim=-1, largest = maximum)
        
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K)
        
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        
        prime = prime.reshape(b, c, -1)

        return prime
        
            
            
    def _prime_N(self, v, qk, K, rand_idx, maximum):
        b, c, t = v.shape

        # Get top-(K-1) indices from the magnitude matrix; shape: [b, t, K-1]
        _, topk_indices = torch.topk(qk, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # Map indices from the sampled space to the full token indices using rand_idx.
        # mapped_tensor will have shape: [b, t, K-1]
        mapped_tensor = rand_idx[topk_indices]

        # Create self indices for each token; shape: [1, t, 1] then expand to [b, t, 1]
        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)

        # Concatenate self index with neighbor indices to form final indices; shape: [b, t, K]
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)

        # Expand final_indices to include the channel dimension; result shape: [b, c, t, K]
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Expand matrix to shape [b, c, t, 1] and then to [b, c, t, K] (ensuring contiguous memory)
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()

        # Gather neighbor features along the token dimension (dim=2)
        prime = torch.gather(v_expanded, dim=2, index=indices_expanded)  # shape: [b, c, t, K]

        # Flatten the token and neighbor dimensions into one: [b, c, t*K]
        prime = prime.reshape(b, c, -1)
        return prime
    
class Conv2d_NN_Attn_spatial(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer for Convolutional Neural Networks.
     - Location Channels : add coordinates -> unshuffle -> flatten -> ConvNN -> unflatten -> shuffle -> remove coordinate 
    
    Attributes: 
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        K (int): Number of Nearest Neighbors for consideration.
        stride (int): Stride size.
        padding (int): Padding size.
        shuffle_pattern (str): Shuffle pattern.
        shuffle_scale (int): Shuffle scale factor.
        samples (int/str): Number of samples to consider.
        magnitude_type (str): Distance or Similarity.
        
    Notes:
        - K must be same as stride. K == stride.
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
                magnitude_type="similarity",
                location_channels=False, 
                image_size=(32, 32)
                ): 
        
        """
        Initializes the Conv2d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
        
        super(Conv2d_NN_Attn_spatial, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples)
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        self.image_size = image_size

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



        self.num_tokens = int((image_size[0] * image_size[1]) / (self.shuffle_scale**2))

        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        
        self.Conv1d_NN_Attn_spatial = Conv1d_NN_Attn_spatial(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples**2,
                                    magnitude_type=self.magnitude_type, 
                                    num_tokens=self.num_tokens
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
                
        # x sample_matrix 
        x_ind = torch.round(torch.linspace(0 + self.padding, x1.shape[2] - self.padding - 1, self.samples)).to(torch.int)
        y_ind = torch.round(torch.linspace(0 + self.padding, x1.shape[3] - self.padding - 1, self.samples)).to(torch.int)
        
        x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
        
        x_idx_flat = x_grid.flatten()
        y_idx_flat = y_grid.flatten()
        
        width = x1.shape[2]
        # flat indices for indexing -> similar to random sampling for ConvNN
        flat_indices = x_idx_flat * width + y_idx_flat
        
        x_sample = self.flatten(x1[:, :, x_grid, y_grid])
        
        # Input Matrix
        x2 = self.flatten(x1)

        x3 = self.Conv1d_NN_Attn_spatial(x2, x_sample, flat_indices.to(x.device))  

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

class Conv1d_NN_Attn_V(nn.Module):
    """
    Convolutional 1D Nearest Neighbors Attention Layer 
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K=3, 
                stride=3, 
                padding=0, 
                shuffle_pattern='N/A', 
                shuffle_scale=1, 
                samples='all', 
                magnitude_type='similarity', 
                num_tokens = 224
                ): 
        
        """
        Initializes the Conv1d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
        
        super(Conv1d_NN_Attn_V, self).__init__()
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride 
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern 
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != 'all' else samples 
        self.magnitude_type = magnitude_type 
        self.maximum = True if self.magnitude_type == 'similarity' else False
        self.num_tokens =  int(num_tokens / 2) if self.shuffle_pattern in ["B", "BA"] else num_tokens        
        
        # Unshuffle layer 
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        
        # Shuffle Layer 
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        
        # Channels for Conv1d Layer
        self.in_channels = in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels

        # Conv1d Layer         
        self.conv1d_layer = Conv1d(in_channels=self.in_channels, 
                                    out_channels=self.out_channels, 
                                    kernel_size=self.K, 
                                    stride=self.stride, 
                                    padding=self.padding)
        
        # Linear Layer for Query, Key, Value
        self.w_v = nn.Linear(self.num_tokens, self.num_tokens, bias=False)

        
        
        
    def forward(self, x): 
        # Consider all samples 
        if self.samples == 'all': 
            # Unshuffle Layer 
            if self.shuffle_pattern in ["B", "BA"]:
                x1 = self.unshuffle_layer(x)
            else:
                x1 = x
            
            # Q, K, V 
            q = x1
            k = x1
            v = self.w_v(x1)
            
            # Calculate Distance/Similarity Matrix + Prime Vmap 2D
            if self.magnitude_type == 'distance': 
                matrix_magnitude = self._calculate_distance_matrix(k, q, sqrt=True)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix(k, q)
                
            prime_2d = self._prime(v, matrix_magnitude, self.K, self.maximum) 
            # Conv1d Layer
            x2 = self.conv1d_layer(prime_2d)
            
            
            # Shuffle Layer 
            if self.shuffle_pattern in ["A", "BA"]:
                x3 = self.shuffle_layer(x2)
            else:
                x3 = x2
            
            return x3
        
        # Consider N samples
        else: 
            # Unshuffle Layer 
            if self.shuffle_pattern in ["B", "BA"]:
                x1 = self.unshuffle_layer(x)
            else:
                x1 = x
                
            # Q, K, V 
            q = x1
            k = x1
            v = self.w_v(x1)    
                
            # Calculate Distance/Similarity Matrix + Prime       
            rand_idx = torch.randperm(x1.shape[2], device=x1.device)[:self.samples]
            
            q_sample = q[:, :, rand_idx]
            
            if self.magnitude_type == 'distance':
                matrix_magnitude = self._calculate_distance_matrix_N(k, q_sample, sqrt=True)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix_N(k, q_sample)
                
            range_idx = torch.arange(len(rand_idx), device=x1.device)
                
        
            if self.magnitude_type == 'distance':
                matrix_magnitude[:, rand_idx, range_idx] = float('inf') 
            elif self.magnitude_type == 'similarity':
                matrix_magnitude[:, rand_idx, range_idx] = float('-inf')
                
            prime = self._prime_N(v, matrix_magnitude, self.K, rand_idx, self.maximum)
            # Conv1d Layer
            x2 = self.conv1d_layer(prime)
            
            # Shuffle Layer
            if self.shuffle_pattern in ["A", "BA"]:
                x3 = self.shuffle_layer(x2)
            else:
                x3 = x2
            
            return x3
        

    def _calculate_similarity_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  # [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  # remove negative values
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  # [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  # remove negative values
        return similarity_matrix
        

    def _calculate_distance_matrix(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True) 
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True) 
        
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        
        # Broadcasting: [B, 1, N] + [B, M, 1] - 2*[B, N, M]
        dist_matrix = norm_squared_K + norm_squared_Q.transpose(2, 1) - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        
        return dist_matrix

    def _calculate_distance_matrix_N(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        
        # Broadcasting: [B, 1, N] + [B, M, 1] - 2*[B, N, M]
        dist_matrix = norm_squared_K + norm_squared_Q - 2 * dot_product
        
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        
        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)
        
        return dist_matrix

    def _prime(self, v, qk, K, maximum):
        b, c, t = v.shape 
        
        _, topk_indices = torch.topk(qk, k=K, dim=-1, largest = maximum)
        
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K)
        
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        
        prime = prime.reshape(b, c, -1)

        return prime
        
            
            
    def _prime_N(self, v, qk, K, rand_idx, maximum):
        b, c, t = v.shape

        # Get top-(K-1) indices from the magnitude matrix; shape: [b, t, K-1]
        _, topk_indices = torch.topk(qk, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # Map indices from the sampled space to the full token indices using rand_idx.
        # mapped_tensor will have shape: [b, t, K-1]
        mapped_tensor = rand_idx[topk_indices]

        # Create self indices for each token; shape: [1, t, 1] then expand to [b, t, 1]
        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)

        # Concatenate self index with neighbor indices to form final indices; shape: [b, t, K]
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)

        # Expand final_indices to include the channel dimension; result shape: [b, c, t, K]
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Expand matrix to shape [b, c, t, 1] and then to [b, c, t, K] (ensuring contiguous memory)
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()

        # Gather neighbor features along the token dimension (dim=2)
        prime = torch.gather(v_expanded, dim=2, index=indices_expanded)  # shape: [b, c, t, K]

        # Flatten the token and neighbor dimensions into one: [b, c, t*K]
        prime = prime.reshape(b, c, -1)
        return prime

class Conv2d_NN_Attn_V(nn.Module): 
    """
    Convolution 2D Nearest Neighbor Layer for Convolutional Neural Networks.
     - Location Channels : add coordinates -> unshuffle -> flatten -> ConvNN -> unflatten -> shuffle -> remove coordinate 
    
    Attributes: 
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        K (int): Number of Nearest Neighbors for consideration.
        stride (int): Stride size.
        padding (int): Padding size.
        shuffle_pattern (str): Shuffle pattern.
        shuffle_scale (int): Shuffle scale factor.
        samples (int/str): Number of samples to consider.
        magnitude_type (str): Distance or Similarity.
        
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K=3,
                stride=3, 
                padding=0, 
                shuffle_pattern="BA", 
                shuffle_scale=2, 
                samples="all", 
                magnitude_type="similarity",
                location_channels=False, 
                image_size=(32, 32)
                ): 
        
        """
        Initializes the Conv2d_NN module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            padding (int): Padding size.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider.
            magnitude_type (str): Distance or Similarity.
        """
        
        super(Conv2d_NN_Attn_V, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != "all" else samples
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



        self.num_tokens = int((image_size[0] * image_size[1]) / (self.shuffle_scale**2))

        self.Conv1d_NN = Conv1d_NN_Attn_V(in_channels=self.in_channels_1d,
                                    out_channels=self.out_channels_1d,
                                    K=self.K,
                                    stride=self.stride,
                                    padding=self.padding,
                                    samples=self.samples, 
                                    shuffle_pattern="NA",
                                    shuffle_scale=1, 
                                    magnitude_type=self.magnitude_type, 
                                    num_tokens=self.num_tokens
                                    )

        self.flatten = nn.Flatten(start_dim=2)
        
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        
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

        x3 = self.Conv1d_NN(x2)  

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
    
class Attention2d(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 shuffle_pattern='BA',
                 shuffle_scale=2,
                 num_heads=4,
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

class PixelShuffle1D(nn.Module): 
    """
    1D Pixel Shuffle Layer for Convolutional Neural Networks.
    
    Attributes: 
        upscale_factor (int): Upscale factor for pixel shuffle. 
        
    Notes:
        Input's channel size must be divisible by the upscale factor. 
    """
    
    def __init__(self, upscale_factor):
        """ 
        Initializes the PixelShuffle1D module.
        
        Parameters:
            upscale_factor (int): Upscale factor for pixel shuffle.
        """
        super(PixelShuffle1D, self).__init__()
        
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
    """
    1D Pixel Unshuffle Layer for Convolutional Neural Networks.
    
    Attributes:
        downscale_factor (int): Downscale factor for pixel unshuffle.
        
    Note:
        Input's token size must be divisible by the downscale factor
    
    """
    
    def __init__(self, downscale_factor):
        """
        Intializes the PixelUnshuffle1D module.
        
        Parameters:
            downscale_factor (int): Downscale factor for pixel unshuffle.
        """
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
    
### Convolutional Nearest Neighbors - Branching Layers ###

class Conv2d_ConvNN_Random_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio=(16, 16), 
                 kernel_size=3,
                 K=9, 
                 shuffle_pattern="BA", 
                 shuffle_scale=2, 
                 samples="all", 
                 magnitude_type="similarity",
                 location_channels = False):
        
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(Conv2d_ConvNN_Random_Branching, self).__init__()
        
        self.in_ch = in_channels 
        self.out_ch = out_channels    
        self.channel_ratio = channel_ratio
        
        self.kernel_size = kernel_size
        self.K = K
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        self.samples = samples
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_ch, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=1),
                nn.ReLU()
            )
        
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN(self.in_ch, 
                          self.channel_ratio[1], 
                          K = self.K, 
                          stride = self.K, 
                          samples = self.samples, 
                          shuffle_pattern=self.shuffle_pattern, 
                          shuffle_scale=self.shuffle_scale,
                          magnitude_type=self.magnitude_type,
                          location_channels = self.location_channels), 
                nn.ReLU()
            )
        
        self.reduce_channels = nn.Conv2d(self.out_ch*2, self.out_ch, 1)

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

class Conv2d_ConvNN_Spatial_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio=(16, 16), 
                 kernel_size=3, 
                 K=9, 
                 samples=8, 
                 shuffle_pattern="BA", 
                 shuffle_scale=2, 
                 magnitude_type="similarity",
                 location_channels = False):
        
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(Conv2d_ConvNN_Spatial_Branching, self).__init__()
        self.kernel_size = kernel_size
        
        self.in_ch = in_channels 
        self.out_ch = out_channels    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        self.samples = samples
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_ch, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=1
                         ),
                nn.ReLU()
            )
            
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_spatial(self.in_ch, 
                                  self.channel_ratio[1], 
                                  K=self.K, 
                                  stride=self.K, 
                                  samples=self.samples, 
                                  shuffle_pattern=self.shuffle_pattern, 
                                  shuffle_scale=self.shuffle_scale,
                                  magnitude_type=self.magnitude_type,
                                  location_channels=self.location_channels), 
                nn.ReLU()
            )

        self.reduce_channels = nn.Conv2d(self.out_ch*2, self.out_ch, 1)

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
        
class Conv2d_ConvNN_Attn_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio=(16, 16), 
                 kernel_size=3, 
                 K=9, 
                 samples="all", 
                 shuffle_pattern="BA", 
                 shuffle_scale=2, 
                 location_channels=False, 
                 magnitude_type="similarity",
                 image_size=(32, 32)):
        
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(Conv2d_ConvNN_Attn_Branching, self).__init__()
        
        self.in_ch = in_channels 
        self.out_ch = out_channels    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        self.samples = samples
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        self.magnitude_type = magnitude_type
        self.location_channels = location_channels
        
        self.image_size = image_size 
        
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_ch, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=1
                         ),
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
                               magnitude_type=self.magnitude_type,
                               location_channels=self.location_channels), 
                nn.ReLU()
            )
        
        self.reduce_channels = nn.Conv2d(self.out_ch*2, self.out_ch, 1)

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
    
class Conv2d_ConvNN_Attn_Spatial_Branching(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 channel_ratio=(16, 16), 
                 kernel_size=3, 
                 K=9, 
                 samples=8, 
                 shuffle_pattern="BA", 
                 shuffle_scale=2,
                 matrix_magnitude="similarity", 
                 location_channels=False, 
                 image_size=(32, 32)):
        
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_ch, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(Conv2d_ConvNN_Attn_Spatial_Branching, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        self.samples = samples
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        self.location_channels = location_channels
        self.matrix_magnitude = matrix_magnitude
        
        self.image_size = image_size 
        
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_ch, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=1
                         ),
                nn.ReLU()
            )
        
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn_spatial(self.in_ch, 
                               self.channel_ratio[1], 
                               K=self.K, 
                               stride=self.K, 
                               samples=self.samples, 
                               shuffle_pattern=self.shuffle_pattern,
                               shuffle_scale=self.shuffle_scale,    
                               image_size=self.image_size, 
                               matrix_magnitude=self.matrix_magnitude,
                               location_channels=self.location_channels), 
                nn.ReLU()
            )
        
        self.reduce_channels = nn.Conv2d(self.out_ch*2, self.out_ch, 1)

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

class Conv2d_ConvNN_Attn_V_Branching(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 channel_ratio=(16, 16), 
                 kernel_size=3, 
                 K=9, 
                 samples="all", 
                 shuffle_pattern="BA",
                 shuffle_scale=2,
                 location_channels = False,
                 image_size = (32, 32)):
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_ch, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(Conv2d_ConvNN_Attn_V_Branching, self).__init__()
        
        self.in_ch = in_ch 
        self.out_ch = out_ch    
        self.channel_ratio = channel_ratio
        self.kernel_size = kernel_size
        self.K = K
        self.samples = samples
        self.shuffle_pattern = shuffle_pattern 
        self.shuffle_scale = shuffle_scale
        self.image_size = image_size
        self.location_channels = location_channels
        
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_ch, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=1
                         ),
                nn.ReLU()
            )
        
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn_V(self.in_ch, 
                                 self.channel_ratio[1], 
                                 K = self.K, 
                                 stride = self.K, 
                                 samples = self.samples, 
                                 shuffle_pattern=self.shuffle_pattern,
                                 shuffle_scale=self.shuffle_scale,
                                 image_size = self.image_size, 
                                 location_channels = self.location_channels
                                ), 
                nn.ReLU()
            )

        self.reduce_channels = nn.Conv2d(self.out_ch*2, self.out_ch, 1)

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

class Attention_ConvNN_Random_Branching(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 channel_ratio=(16, 16), 
                 K=9, 
                 shuffle_pattern="BA",
                 shuffle_scale=2,
                 num_heads=4, 
                 samples = "all", 
                 matrix_magnitude="similarity",
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
        self.matrix_magnitude = matrix_magnitude
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
                          matrix_magnitude=self.matrix_magnitude,
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
                 shuffle_pattern="BA",  
                 shuffle_scale=2,    
                 num_heads=4,   
                 matrix_magnitude="similarity",
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
        
        self.matrix_magnitude = matrix_magnitude
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
                                  matrix_magnitude=self.matrix_magnitude,
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
                 shuffle_pattern="BA", 
                 shuffle_scale=2, 
                 num_heads=4, 
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
    
class Attention_ConvNN_Attn_Spatial_Branching(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 channel_ratio=(16, 16), 
                 kernel_size=3, 
                 K=9, 
                 samples=8, 
                 shuffle_pattern="BA", 
                 shuffle_scale=2, 
                 num_heads=4, 
                 location_channels = False, 
                 matrix_magnitude="similarity",
                 image_size=(32, 32)):
        
        # Channel_ratio must add up to 2*out_ch
        assert sum(channel_ratio) == 2*out_ch, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        super(Attention_ConvNN_Attn_Spatial_Branching, self).__init__()
        
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
        
        self.matrix_magnitude = matrix_magnitude
        self.location_channels = location_channels
        
        
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_ch, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            matrix_magnitude=self.matrix_magnitude,
                            location_channels=self.location_channels),
                nn.ReLU()
            )
            
        
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn_spatial(self.in_ch, 
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
                    shuffle_pattern="BA", 
                    shuffle_scale=2, 
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
                 shuffle_pattern="BA", 
                 shuffle_scale=2, 
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
    

'''Model Modules'''
class B_Conv2d_ConvNN_K_All(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, kernel_size=3, channel_ratio=(16, 16), num_classes=100, device="mps"):
        super(B_Conv2d_ConvNN_K_All, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Conv2d_ConvNN_Random_Branching(in_ch, mid_ch, K=K, samples='all', channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Conv2d_ConvNN_Random_Branching(mid_ch, mid_ch, K=K, samples='all', channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_K_All" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Conv2d_ConvNN_K_N(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, kernel_size=3, channel_ratio=(16, 16), num_classes=100, device="mps"):
        super(B_Conv2d_ConvNN_K_N, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Conv2d_ConvNN_Random_Branching(in_ch, mid_ch, K=K, samples=N, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Conv2d_ConvNN_Random_Branching(mid_ch, mid_ch, K=K, samples=N, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_K_N" # Renamed for clarity

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Conv2d_ConvNN_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, kernel_size=3, channel_ratio=(16, 16), num_classes=100, device="mps"):
        super(B_Conv2d_ConvNN_Spatial_K_N, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Conv2d_ConvNN_Spatial_Branching(in_ch, mid_ch, K=K, samples=N, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Conv2d_ConvNN_Spatial_Branching(mid_ch, mid_ch, K=K, samples=N, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_Spatial_K_N" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Conv2d_ConvNN_Attn_K_N(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, kernel_size=3, channel_ratio=(16, 16),num_classes=100, device="mps"):
        super(B_Conv2d_ConvNN_Attn_K_N, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Conv2d_ConvNN_Attn_Branching(in_ch, mid_ch, K=K, samples=N, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Conv2d_ConvNN_Attn_Branching(mid_ch, mid_ch, K=K, samples=N, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_Attn_K_N" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Conv2d_ConvNN_Attn_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, kernel_size=3, channel_ratio=(16, 16),num_classes=100, device="mps"):
        super(B_Conv2d_ConvNN_Attn_Spatial_K_N, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Conv2d_ConvNN_Attn_Spatial_Branching(in_ch, mid_ch, K=K, samples=N, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Conv2d_ConvNN_Attn_Spatial_Branching(mid_ch, mid_ch, K=K, samples=N, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_Attn_Spatial_K_N" 


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Conv2d_ConvNN_Attn_V_K_N(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, kernel_size=3, channel_ratio=(16, 16),num_classes=100, device="mps"):
        super(B_Conv2d_ConvNN_Attn_V_K_N, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Conv2d_ConvNN_Attn_V_Branching(in_ch, mid_ch, K=K, samples=N, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Conv2d_ConvNN_Attn_V_Branching(mid_ch, mid_ch, K=K, samples=N, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Conv2d_ConvNN_Attn_V_K_N" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Attention_ConvNN_K_All(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, num_heads=4,  channel_ratio=(16, 16),num_classes=100, device="mps"):
        super(B_Attention_ConvNN_K_All, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Attention_ConvNN_Random_Branching(in_ch, mid_ch, K=K, samples='all', num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Attention_ConvNN_Random_Branching(mid_ch, mid_ch, K=K, samples='all', num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_K_All" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Attention_ConvNN_K_N(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, num_heads=4, channel_ratio=(16, 16),num_classes=100, device="mps"):
        super(B_Attention_ConvNN_K_N, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Attention_ConvNN_Random_Branching(in_ch, mid_ch, K=K, samples=N, num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Attention_ConvNN_Random_Branching(mid_ch, mid_ch, K=K, samples=N, num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_K_N" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Attention_ConvNN_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, channel_ratio=(16, 16), num_heads=4, num_classes=100, device="mps"):
        super(B_Attention_ConvNN_Spatial_K_N, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Attention_ConvNN_Spatial_Branching(in_ch, mid_ch, K=K,  samples=N, num_heads=num_heads,  channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Attention_ConvNN_Spatial_Branching(mid_ch, mid_ch, K=K, samples=N, num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_Spatial_K_N" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Attention_ConvNN_Attn_K_N(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, num_heads=4, channel_ratio=(16, 16),num_classes=100, device="mps"):
        super(B_Attention_ConvNN_Attn_K_N, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Attention_ConvNN_Attn_Branching(in_ch, mid_ch, K=K, samples=N, num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Attention_ConvNN_Attn_Branching(mid_ch, mid_ch, K=K, samples=N, num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_Attn_K_N" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)
            
class B_Attention_ConvNN_Attn_Spatial_K_N(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, num_heads=4, channel_ratio=(16, 16),num_classes=100, device="mps"):
        super(B_Attention_ConvNN_Attn_Spatial_K_N, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Attention_ConvNN_Attn_Spatial_Branching(in_ch, mid_ch, K=K, samples=N, num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Attention_ConvNN_Attn_Spatial_Branching(mid_ch, mid_ch, K=K, samples=N, num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_Attn_Spatial_K_N" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Attention_ConvNN_Attn_V_K_N(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, num_heads=4, channel_ratio=(16, 16),num_classes=100, device="mps"):
        super(B_Attention_ConvNN_Attn_V_K_N, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Attention_ConvNN_Attn_V_Branching(in_ch, mid_ch, K=K, samples=N, num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Attention_ConvNN_Attn_V_Branching(mid_ch, mid_ch, K=K, samples=N, num_heads=num_heads, channel_ratio=channel_ratio))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Attention_ConvNN_Attn_V_K_N" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

class B_Attention_Conv2d(nn.Module):
    def __init__(self, in_ch=3, mid_ch=16, num_layers=2, kernel_size=3, num_heads=4, channel_ratio=(16, 16), num_classes=100, device="mps"):
        super(B_Attention_Conv2d, self).__init__()
        
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert mid_ch >= 8, "Middle channels must be at least 8"
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(Attention_Conv2d_Branching(in_ch, mid_ch, num_heads=num_heads, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())
            else: 
                layers.append(Attention_Conv2d_Branching(mid_ch, mid_ch,num_heads=num_heads, channel_ratio=channel_ratio, kernel_size=kernel_size))
                # layers.append(nn.BatchNorm2d(mid_ch)) 
                layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        
        flattened_size = mid_ch * 32 * 32
        
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(flattened_size, num_classes), # Increased intermediate size
        )

        self.device = device
        self.to(self.device)
        self.name = "B_Attention_Conv2d" # Renamed for clarity


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self, input_size = (3, 32, 32)): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but input_size doesn't include it
            summary(self, input_size=input_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)

'''Register Models'''

@register_model
def b_conv2d_convnn_k_all_100(pretrained=False, **kwargs):
    """Branching ConvNN K All Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_K_All(in_ch=3, mid_ch=16, num_layers=2, K=9, kernel_size=3, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_k_all_10(pretrained=False, **kwargs):
    """Branching ConvNN K All Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_K_All(in_ch=3, mid_ch=16, num_layers=2, K=9, kernel_size=3, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_k_n_100(pretrained=False, **kwargs):
    """Branching ConvNN K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_K_N(in_ch=3, K=9, N=64, kernel_size=3, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_k_n_10(pretrained=False, **kwargs):
    """Branching ConvNN K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, kernel_size=3, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_spatial_k_n_100(pretrained=False, **kwargs):
    """Branching ConvNN Spatial K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_Spatial_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, kernel_size=3, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_spatial_k_n_10(pretrained=False, **kwargs):
    """Branching ConvNN Spatial K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_Spatial_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, kernel_size=3, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_attn_k_n_100(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_Attn_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, kernel_size=3, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_attn_k_n_10(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_Attn_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, kernel_size=3, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_attn_spatial_k_n_100(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_Attn_Spatial_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, kernel_size=3, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_attn_spatial_k_n_10(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_Attn_Spatial_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, kernel_size=3, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_attn_v_k_n_100(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_Attn_V_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, kernel_size=3, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_conv2d_convnn_attn_v_k_n_10(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Conv2d_ConvNN_Attn_V_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, kernel_size=3, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model


@register_model
def b_attention_convnn_k_all_100(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_K_All(in_ch=3, mid_ch=16, num_layers=2, K=9, num_heads=4, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_k_all_10(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_K_All(in_ch=3, mid_ch=16, num_layers=2, K=9, num_heads=4, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_k_n_100(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, num_heads=4, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_k_n_10(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, num_heads=4, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_spatial_k_n_100(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_Spatial_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, num_heads=4,  channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_spatial_k_n_10(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_Spatial_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, num_heads=4,  channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_attn_k_n_100(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_Attn_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, num_heads=4, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_attn_k_n_10(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_Attn_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, num_heads=4, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_attn_spatial_k_n_100(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_Attn_Spatial_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, num_heads=4, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_attn_spatial_k_n_10(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_Attn_Spatial_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=8, num_heads=4, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_attn_v_k_n_100(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_Attn_V_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, num_heads=4,  channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_convnn_attn_v_k_n_10(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_ConvNN_Attn_V_K_N(in_ch=3, mid_ch=16, num_layers=2, K=9, N=64, num_heads=4, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_conv2d_100(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_Conv2d(in_ch=3, mid_ch=16, num_layers=2, kernel_size=3, num_heads=4, channel_ratio=(16, 16), num_classes=100, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

@register_model
def b_attention_conv2d_10(pretrained=False, **kwargs):
    """Branching ConvNN Attention K N Samples"""
    # Ensure this function now uses the updated CNN class
    model = B_Attention_Conv2d(in_ch=3, mid_ch=16, num_layers=2, kernel_size=3, num_heads=4, channel_ratio=(16, 16), num_classes=10, device='mps')
    
    # model.default_cfg = _cfg(
    #     input_size=(3, 32, 32),
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     num_classes=100
    # )
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test and parameter count comparison
    models = [
        b_conv2d_convnn_k_all_100(), 
        b_conv2d_convnn_k_n_100(), 
        b_conv2d_convnn_spatial_k_n_100(),
        b_conv2d_convnn_attn_k_n_100(),
        b_conv2d_convnn_attn_spatial_k_n_100(),
        b_conv2d_convnn_attn_v_k_n_100(),
        b_attention_convnn_k_all_100(),
        b_attention_convnn_k_n_100(),
        b_attention_convnn_spatial_k_n_100(),
        b_attention_convnn_attn_k_n_100(),
        b_attention_convnn_attn_spatial_k_n_100(),
        b_attention_convnn_attn_v_k_n_100(),
        b_attention_conv2d_100(),
        
        b_conv2d_convnn_k_all_10(), 
        b_conv2d_convnn_k_n_10(), 
        b_conv2d_convnn_spatial_k_n_10(),
        b_conv2d_convnn_attn_k_n_10(),
        b_conv2d_convnn_attn_spatial_k_n_10(), 
        b_conv2d_convnn_attn_v_k_n_10(),
        b_attention_convnn_k_all_10(),
        b_attention_convnn_k_n_10(),
        b_attention_convnn_spatial_k_n_10(),
        b_attention_convnn_attn_k_n_10(),
        b_attention_convnn_attn_spatial_k_n_10(), 
        b_attention_convnn_attn_v_k_n_10(),
        b_attention_conv2d_10()
        
    ]
    
    x = torch.randn(1, 3, 32, 32).to('mps')

    for model in models:
        print(f"Model: {model.name}")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Output shape: {model(x).shape}")
        print("-" * 30)
    
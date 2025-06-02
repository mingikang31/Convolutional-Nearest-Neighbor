"""Supplemental 1D layers for 2D Layers"""

"""
# Standalone Layers
(1) Conv1d_NN
(2) Conv1d_NN_spatial 
(3) Conv1d_NN_Attn
(4) Conv1d_NN_Attn_spatial
(5) Conv1d_NN_Attn_V
(6) Attention1d

(*) PixelShuffle1D
(*) PixelUnshuffle1D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""(1) Conv1d_NN"""
class Conv1d_NN(nn.Module): 
    """
    Convolution 1D Nearest Neighbor Layer
    
    Notes:
        - K must be same as stride. K == stride.
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 K,
                 stride, 
                 padding, 
                 shuffle_pattern, 
                 shuffle_scale, 
                 samples, 
                 magnitude_type
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
        
        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'" 
        
        # Initialize parameters
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
        
        # Shuffle1D/Unshuffle1D Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
                
        # Adjust Channels for PixelShuffle
        self.in_channels = in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels

        # Conv1d Layer 
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
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
    
"""(2) Conv1d_NN_spatial"""
class Conv1d_NN_Spatial(nn.Module): 
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
        This is not a standalone module. It is used as a part of the Conv2d_NN_Spatial module.
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 K, 
                 stride, 
                 padding, 
                 magnitude_type
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
        super(Conv1d_NN_Spatial, self).__init__()
        
        ### Assertions ###
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"

        # Initialize parameters
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
        batched_process = torch.vmap(Conv1d_NN_Spatial.process_batch_N, in_dims=(0, 0, None, None), out_dims=0)
        prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, spatial_idx, flatten=True, maximum=maximum)
        return prime 
    
    @staticmethod
    def prime_vmap_3d_N(matrix, magnitude_matrix, num_nearest_neighbors, spatial_idx, maximum): 
        """Vectorization / Vmap Implementation for Nearest Neighbor Tensor 3D"""
        batched_process = torch.vmap(Conv1d_NN_Spatial.process_batch_N, in_dims=(0, 0, None, None), out_dims=0)
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
  
"""(3) Conv1d_NN_Attn"""
class Conv1d_NN_Attn(nn.Module):
    """
    Convolutional 1D Nearest Neighbors Attention Layer 
    """
    
    def __init__(self, 
                in_channels, 
                out_channels, 
                K,
                stride, 
                padding, 
                shuffle_pattern, 
                shuffle_scale, 
                samples, 
                num_tokens,
                magnitude_type
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
        
        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"        
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride 
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern 
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != 'all' else samples 
        self.num_tokens =  int(num_tokens / 2) if self.shuffle_pattern in ["B", "BA"] else num_tokens        
        
        self.magnitude_type = magnitude_type 
        self.maximum = True if self.magnitude_type == 'similarity' else False
        
        # Shuffle1D/Unshuffle1D Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for Shuffle
        self.in_channels = in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels

        # Conv1d Layer 
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                    out_channels=self.out_channels, 
                                    kernel_size=self.K, 
                                    stride=self.stride, 
                                    padding=self.padding)
        
        # Linear Layer for Query, Key, Value
        self.w_q = nn.Linear(self.num_tokens, self.num_tokens, bias=False)
        self.w_k = nn.Linear(self.num_tokens, self.num_tokens, bias=False)
        self.w_v = nn.Linear(self.num_tokens, self.num_tokens, bias=False)
        self.w_o = nn.Linear(self.num_tokens, self.num_tokens, bias=False)
        
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
            
            x4 = self.w_o(x3)
            return x4
        
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
            
            x4 = self.w_o(x3)
            return x4

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
  
"""(4) Conv1d_NN_Attn_spatial"""
class Conv1d_NN_Attn_Spatial(nn.Module):
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
                num_tokens,
                magnitude_type, 
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
        super(Conv1d_NN_Attn_Spatial, self).__init__()
        
        ### Assertions ###
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert samples < num_tokens, "Error: samples must be less than num_tokens"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride 
        self.padding = padding
        self.samples = samples 
        self.num_tokens = num_tokens        

        self.magnitude_type = magnitude_type 
        self.maximum = True if self.magnitude_type == 'similarity' else False
        
        # Conv1d Layer 
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
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
    
"""(5) Conv1d_NN_Attn_V"""
class Conv1d_NN_Attn_V(nn.Module):
    """
    Convolutional 1D Nearest Neighbors Attention Layer 
    """
    def __init__(self, 
                in_channels, 
                out_channels, 
                K,
                stride, 
                padding, 
                shuffle_pattern, 
                shuffle_scale, 
                samples, 
                num_tokens,
                magnitude_type, 
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
            num_tokens (int): Number of tokens for the linear layer.
        """
        super(Conv1d_NN_Attn_V, self).__init__()

        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert samples == "all" or (isinstance(samples, int) and samples > 0), "Error: samples must be greater than 0 or 'all'"        
        
        # Initialize parameters    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride 
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern 
        self.shuffle_scale = shuffle_scale
        self.samples = int(samples) if samples != 'all' else samples 
        self.num_tokens =  int(num_tokens / 2) if self.shuffle_pattern in ["B", "BA"] else num_tokens        

        self.magnitude_type = magnitude_type 
        self.maximum = True if self.magnitude_type == 'similarity' else False

        # Shuffle1D/Unshuffle1D Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for Shuffle
        self.in_channels = in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels

        # Conv1d Layer         
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                    out_channels=self.out_channels, 
                                    kernel_size=self.K, 
                                    stride=self.stride, 
                                    padding=self.padding)
        
        # Linear Layer for Value
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
    
"""(6) Attention1d"""
class Attention1d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_heads,
                 shuffle_pattern,
                 shuffle_scale 
                 ):
        super(Attention1d, self).__init__()
        
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert shuffle_scale > 0, "Error: shuffle_scale must be greater than 0"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        # Shuffle1D/Unshuffle1D Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for Shuffle
        self.in_channels = self.in_channels * self.shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = self.out_channels * self.shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels
        
        # MultiHead Attention Layer
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.out_channels, num_heads=self.num_heads, batch_first=True)
        
        # 1x1 Convolution Layer
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
    
"""(*) PixelShuffle1D"""
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

"""(*) PixelUnshuffle1D"""
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
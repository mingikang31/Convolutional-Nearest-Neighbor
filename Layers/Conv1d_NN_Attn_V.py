'''Convolution 1D Nearest Neighbors Attention Layer V'''

'''
Conv1d_NN_Attn_V is a variation of Conv1d_NN that incorporates the attention mechanism. 

- The only difference is that we only run the V matrix into a linear layer instead of all K, V, and Q. 
'''

import torch 
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F 
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D
import numpy as np 

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
    
    
def example_usage():
    ex = torch.randn(32, 3, 32)

    conv1d_NN_attn = Conv1d_NN_Attn_V(in_channels=3,
                                    out_channels=3,
                                    K=3,
                                    stride=3,
                                    padding=0,
                                    shuffle_pattern='NA',
                                    shuffle_scale=2,
                                    samples=10,
                                    magnitude_type='similarity', 
                                    num_tokens=32)
    out = conv1d_NN_attn(ex)
    print('Conv1d_NN_Attn_V output shape:', out.shape)

if __name__ == "__main__":
    example_usage()
    

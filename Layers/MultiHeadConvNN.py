'''Multihead Convolutional Nearest Neighbor Attention Layer'''

'''
This module implements a multi-head convolutional nearest neighbor attention layer, that is used in transformer architectures. 

- The linear projects for query, key, and value are for the channel dimension of the input tensor.
'''
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class MultiHeadConvNN(nn.Module):
    def __init__(self, d_model, num_heads, K, samples, magnitude_type):
        super(MultiHeadConvNN, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.K = K
        self.samples = int(samples) if samples > 0 else None
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False
        
        # Linear projections for query, key, value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)   
        
        
        self.in_channels = d_model // num_heads
        self.out_channels = d_model // num_heads
        self.kernel_size = K
        self.stride = K
        
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )
        
    def split_head(self, x): 
        batch_size, seq_length, d_model = x.size()
        self.batch_size = batch_size
        self.seq_length = seq_length
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) 
    
    def batch_split(self, x): 
        x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
        return x.permute(0, 1, 3, 2).contiguous()
        
    def batch_combine(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)
        
    def forward(self, x):
        if self.samples is None: # All Samples
            print("All Samples")
            q = self.batch_combine(self.split_head(self.W_q(x)))
            k = self.batch_combine(self.split_head(self.W_k(x)))
            v = self.batch_combine(self.split_head(self.W_v(x)))
            
            
            # Calculate Distance/Similarity Matrix + Prime Vmap 2D
            if self.magnitude_type == 'distance': 
                matrix_magnitude = self._calculate_distance_matrix(k, q, sqrt=True)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix(k, q)
                
            prime = self._prime(v, matrix_magnitude, self.K, self.maximum) 
            x = self.conv(prime)  
            
            x = self.W_o(self.combine_heads(self.batch_split(x.permute(0, 2, 1))))
      
            return x
        
        else: # Random Samples
            q = self.batch_combine(self.split_head(self.W_q(x)))
            k = self.batch_combine(self.split_head(self.W_k(x)))
            v = self.batch_combine(self.split_head(self.W_v(x)))
            
            # Calculate Distance/Similarity Matrix + Prime       
            rand_idx = torch.randperm(q.shape[2], device=q.device)[:self.samples]
            
            q_sample = q[:, :, rand_idx]
            
            if self.magnitude_type == 'distance':
                matrix_magnitude = self._calculate_distance_matrix_N(k, q_sample, sqrt=True)
            elif self.magnitude_type == 'similarity':
                matrix_magnitude = self._calculate_similarity_matrix_N(k, q_sample)
                
            range_idx = torch.arange(len(rand_idx), device=q.device)
                
        
            if self.magnitude_type == 'distance':
                matrix_magnitude[:, rand_idx, range_idx] = float('inf') 
            elif self.magnitude_type == 'similarity':
                matrix_magnitude[:, rand_idx, range_idx] = float('-inf')
            
            
            prime = self._prime_N(v, matrix_magnitude, self.K, rand_idx, self.maximum)
            
            # Conv1d Layer
            x = self.conv(prime)  
            
            x = self.W_o(self.combine_heads(self.batch_split(x.permute(0, 2, 1))))
      
            return x        
    
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

        _, topk_indices = torch.topk(qk, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        mapped_tensor = rand_idx[topk_indices]

        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)

        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)

        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()

        prime = torch.gather(v_expanded, dim=2, index=indices_expanded)

        prime = prime.reshape(b, c, -1)
        return prime
           
if __name__ == "__main__":
    x = torch.randn(128, 196, 6)

    multiheadconvnn = MultiHeadConvNN(
        d_model=6, 
        num_heads=3, 
        K=3, 
        samples=35, 
        magnitude_type='similarity')
    output = multiheadconvnn(x)
    print("Input shape:", x.shape) 
    print("Output shape:", output.shape) 
    
    


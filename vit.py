'''ViT Model with Conv2d, ConvNN, ConvNN_Attn, Attention'''

# Torch Imports 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchsummary import summary 
import numpy as np 

from typing import cast, Union

'''VGG Model Class'''
class ViT(nn.Module): 
    def __init__(self, args): 
        super(ViT, self).__init__()
        assert args.img_size[1] % args.patch_size == 0 and args.img_size[2] % args.patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert args.d_hidden % args.num_heads == 0, "d_hidden must be divisible by n_heads"
        
        self.args = args
        self.args.model = "VIT"
        self.model = "VIT"
        
        self.d_hidden = self.args.d_hidden 
        self.d_mlp = self.args.d_mlp
        
        self.img_size = self.args.img_size[1:]
        self.n_classes = self.args.num_classes # Number of Classes
        self.n_heads = self.args.num_heads
        self.patch_size = (self.args.patch_size, self.args.patch_size) # Patch Size
        self.n_channels = self.args.img_size[0]
        self.n_layers = self.args.num_layers # Number of Layers
        
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        
        self.dropout = self.args.dropout # Dropout Rate
        self.attention_dropout = self.args.attention_dropout # Attention Dropout Rate   
        self.max_seq_length = self.n_patches + 1 # +1 for class token
        
        self.patch_embedding = PatchEmbedding(self.d_hidden, self.img_size, self.patch_size, self.n_channels) # Patch Embedding Layer
        self.positional_encoding = PositionalEncoding(self.d_hidden, self.max_seq_length)
        
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(
            args=args, 
            d_hidden=self.d_hidden, 
            d_mlp=self.d_mlp, 
            num_heads=self.n_heads, 
            dropout=self.dropout, 
            attention_dropout=self.attention_dropout
            ) for _ in range(self.n_layers)])
        
        self.classifier = nn.Linear(self.d_hidden, self.n_classes)
        
        self.device = args.device
        
        self.to(self.device)
        self.name = f"{self.args.model} {self.args.layer}"
        
    def forward(self, x): 
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0]) # Taking the CLS token for classification
        return x

    def summary(self): 
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            summary(self, input_size=self.img_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            self.to(original_device)
        
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
        
class PatchEmbedding(nn.Module): 
    def __init__(self, d_hidden, img_size, patch_size, n_channels=3): 
        super(PatchEmbedding, self).__init__()
        
        self.d_hidden = d_hidden # Dimensionality of Model 
        self.img_size = img_size # Size of Image
        self.patch_size = patch_size # Patch Size 
        self.n_channels = n_channels # Number of Channels in Image
        
        self.linear_projection = nn.Conv2d(in_channels=n_channels, out_channels=d_hidden, kernel_size=patch_size, stride=patch_size) # Linear Projection Layer
        self.norm = nn.LayerNorm(d_hidden) # Normalization Layer
        
        self.flatten = nn.Flatten(start_dim=2)
        
    def forward(self, x): 
        x = self.linear_projection(x) # (B, C, H, W) -> (B, d_hidden, H', W')
        x = self.flatten(x) # (B, d_hidden, H', W') -> (B, d_hidden, n_patches)
        x = x.transpose(1, 2) # (B, d_hidden, n_patches) -> (B, n_patches, d_hidden)
        x = self.norm(x) # (B, n_patches, d_hidden) -> (B, n_patches, d_hidden)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_hidden, max_seq_length): 
        super(PositionalEncoding, self).__init__()
        
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, d_hidden)) # Classification Token

        pe = torch.zeros(max_seq_length, d_hidden)  # Positional Encoding Tensor
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_hidden, 2).float() * (-np.log(10000.0) / d_hidden))  

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): 
        # Expand to have class token for each image in batch 
        tokens_batch = self.cls_tokens.expand(x.shape[0], -1, -1) # (B, 1, d_hidden)
        
        # Concatenate class token with positional encoding
        x = torch.cat((tokens_batch, x), dim=1)
        
        # Add positional encoding to the input 
        x = x + self.pe[:, :x.size(1)].to(x.device) 
        return x

class TransformerEncoder(nn.Module): 
    def __init__(self, args, d_hidden, d_mlp, num_heads, dropout, attention_dropout):
        super(TransformerEncoder, self).__init__()
        self.args = args 

        self.d_hidden = d_hidden 
        self.d_mlp = d_mlp
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        
        if args.layer == "Attention":
            self.attention = MultiHeadAttention(d_hidden, num_heads, attention_dropout)
        elif args.layer == "ConvNN": 
            self.attention = MultiHeadConvNN(d_hidden, num_heads, args.K, args.sampling_type, args.num_samples, args.sample_padding, args.magnitude_type)
        elif args.layer == "ConvNNAttention": 
            self.attention = MultiHeadConvNNAttention(d_hidden, num_heads, args.K, args.sampling_type, args.num_samples, args.sample_padding, args.magnitude_type)
        elif args.layer == "Conv1d":
            self.attention = MultiHeadConv1d(d_hidden, num_heads, args.kernel_size)
        elif args.layer == "Conv1dAttention":
            self.attention = MultiHeadConv1dAttention(d_hidden, num_heads, args.kernel_size)
        elif args.layer == "KvtAttention":
            self.attention = MultiHeadKvtAttention(
                                        dim=d_hidden, 
                                        num_heads=num_heads, 
                                        attn_drop=attention_dropout,
                                        topk=args.K) 
        
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Multilayer Perceptron 
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_hidden)
        )
        
    def forward(self, x): 
        # Pre-Norm Multi-Head Attention 
        norm_x = self.norm1(x) 
        attn_output = self.attention(norm_x)  
        x = x + self.dropout1(attn_output)
        
        # Post-Norm Feed Forward Network
        norm_x = self.norm2(x)  
        mlp_output = self.mlp(norm_x)
        x = x + self.dropout2(mlp_output)  
        return x

"""Multi-Head Layers for Transformer Encoder"""
class MultiHeadAttention(nn.Module): 
    def __init__(self, d_hidden, num_heads, attention_dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
        
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_k = d_hidden // num_heads # dimension of each head
        self.dropout = nn.Dropout(attention_dropout)
        
        self.W_q = nn.Linear(d_hidden, d_hidden)
        self.W_k = nn.Linear(d_hidden, d_hidden)
        self.W_v = nn.Linear(d_hidden, d_hidden)
        self.W_o = nn.Linear(d_hidden, d_hidden)        
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
    
    def split_head(self, x): 
        batch_size, seq_length, d_hidden = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden) 
    
    def forward(self, x, mask=None):
        q = self.split_head(self.W_q(x)) # (B, num_heads, seq_length, d_k)
        k = self.split_head(self.W_k(x))
        v = self.split_head(self.W_v(x))
        
        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask) # (B, num_heads, seq_length, d_k)
        output = self.W_o(self.combine_heads(attn_output)) # (B, seq_length, d_hidden)
        return output

class MultiHeadConvNNAttention(nn.Module):
    def __init__(self, d_hidden, num_heads, K, sampling_type, num_samples, sample_padding, magnitude_type):
        super(MultiHeadConvNNAttention, self).__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_k = d_hidden // num_heads
        
        self.K = K
        self.sampling_type = sampling_type
        self.sampling_type = sampling_type
        self.num_samples = num_samples if num_samples != -1 else 'all' # -1 for all samples 
        self.sample_padding = sample_padding if sampling_type == 'spatial' else 0    
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False
        
        # Linear projections for query, key, value
        self.W_q = nn.Linear(d_hidden, d_hidden)
        self.W_k = nn.Linear(d_hidden, d_hidden)
        self.W_v = nn.Linear(d_hidden, d_hidden)
        self.W_o = nn.Linear(d_hidden, d_hidden)   
        
        
        self.in_channels = d_hidden // num_heads
        self.out_channels = d_hidden // num_heads
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
        batch_size, seq_length, d_hidden = x.size()
        self.batch_size = batch_size
        self.seq_length = seq_length
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden) 
    
    def batch_split(self, x): 
        x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
        return x.permute(0, 1, 3, 2).contiguous()
        
    def batch_combine(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)
        
    def forward(self, x):
        k = self.batch_combine(self.split_head(self.W_k(x)))
        v = self.batch_combine(self.split_head(self.W_v(x)))

        
        if self.sampling_type == 'all': # All Samples
            q = self.batch_combine(self.split_head(self.W_q(x)))
            matrix_magnitude = self._calculate_distance_matrix(k, q, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix(k, q)
                
            prime = self._prime(v, matrix_magnitude, self.K, self.maximum) 

        elif self.sampling_type == 'random': # Random Samples
            q = self.batch_combine(self.split_head(self.W_q(x)))
            
            # Select random samples from q
            rand_idx = torch.randperm(q.shape[2], device=q.device)[:self.num_samples]
            q_sample = q[:, :, rand_idx]

            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix_N(k, q_sample, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(k, q_sample)
            range_idx = torch.arange(len(rand_idx), device=q.device)
            matrix_magnitude[:, rand_idx, range_idx] = float('inf') if self.magnitude_type == 'distance' else float('-inf')
            prime = self._prime_N(v, matrix_magnitude, self.K, rand_idx, self.maximum)

        elif self.sampling_type == 'spatial': # Spatial Samples
            q = self.batch_combine(self.split_head(self.W_q(x)))

            # Select spatial samples from q
            spat_idx = torch.linspace(0 + self.sample_padding, q.shape[2] - self.sample_padding - 1, self.num_samples, device=q.device).long()
            q_sample = q[:, :, spat_idx]

            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix_N(k, q_sample, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(k, q_sample)
            range_idx = torch.arange(len(spat_idx), device=q.device)
            matrix_magnitude[:, spat_idx, range_idx] = float('inf') if self.magnitude_type == 'distance' else float('-inf')
            prime = self._prime_N(v, matrix_magnitude, self.K, spat_idx, self.maximum)
        else: 
            raise ValueError("Invalid sampling_type. Must be one of ['all', 'random', 'spatial']")

        x = self.conv(prime)  
        x = self.W_o(self.combine_heads(self.batch_split(x.permute(0, 2, 1))))
        return x       

    def _calculate_similarity_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm) 
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm) 
        similarity_matrix = torch.clamp(similarity_matrix, min=0) 
        return similarity_matrix

    def _calculate_distance_matrix(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True) 
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True) 
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        dist_matrix = norm_squared_K + norm_squared_Q.transpose(2, 1) - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix
        return dist_matrix

    def _calculate_distance_matrix_N(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        dist_matrix = norm_squared_K + norm_squared_Q - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix
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

class MultiHeadConvNN(nn.Module):
    def __init__(self, d_hidden, num_heads, K, sampling_type, num_samples, sample_padding, magnitude_type, seq_length=197):
        super(MultiHeadConvNN, self).__init__() 
        
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"   
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_k = d_hidden // num_heads
        
        self.K = K
        self.sampling_type = sampling_type
        self.sampling_type = sampling_type
        self.num_samples = num_samples if num_samples != -1 else 'all' # -1 for all samples 
        self.sample_padding = sample_padding if sampling_type == 'spatial' else 0    
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False
        self.seq_length = seq_length
        
        # Linear projections for query, key, value
        self.W_q = nn.Linear(self.seq_length, self.seq_length)
        self.W_k = nn.Linear(self.seq_length, self.seq_length)
        self.W_v = nn.Linear(self.seq_length, self.seq_length)
        self.W_o = nn.Linear(self.seq_length, self.seq_length)
        
        self.in_channels = d_hidden // num_heads
        self.out_channels = d_hidden // num_heads
        self.kernel_size = K
        self.stride = K
        self.padding = 0 
        
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def split_head(self, x):
        batch_size, d_hidden, seq_length = x.size()
        self.batch_size = batch_size
        self.seq_length = seq_length
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) 
    
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, self.d_hidden, seq_length) 
    
    def batch_split(self, x): 
        x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
        return x.permute(0, 1, 3, 2).contiguous()
        
    def batch_combine(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)

    def forward(self, x):
        x = x.permute(0, 2, 1) 

        k = self.batch_combine(self.split_head(self.W_k(x)))
        v = self.batch_combine(self.split_head(self.W_v(x)))

        
        if self.sampling_type == 'all': # All Samples
            q = self.batch_combine(self.split_head(self.W_q(x)))
            matrix_magnitude = self._calculate_distance_matrix(k, q, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix(k, q)
                
            prime = self._prime(v, matrix_magnitude, self.K, self.maximum) 

        elif self.sampling_type == 'random': # Random Samples
            q = self.batch_combine(self.split_head(self.W_q(x)))
            
            # Select random samples from q
            rand_idx = torch.randperm(q.shape[2], device=q.device)[:self.num_samples]
            q_sample = q[:, :, rand_idx]

            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix_N(k, q_sample, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(k, q_sample)
            range_idx = torch.arange(len(rand_idx), device=q.device)
            matrix_magnitude[:, rand_idx, range_idx] = float('inf') if self.magnitude_type == 'distance' else float('-inf')
            prime = self._prime_N(v, matrix_magnitude, self.K, rand_idx, self.maximum)

        elif self.sampling_type == 'spatial': # Spatial Samples
            q = self.batch_combine(self.split_head(self.W_q(x)))

            # Select spatial samples from q
            spat_idx = torch.linspace(0 + self.sample_padding, q.shape[2] - self.sample_padding - 1, self.num_samples, device=q.device).long()
            q_sample = q[:, :, spat_idx]

            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix_N(k, q_sample, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(k, q_sample)
            range_idx = torch.arange(len(spat_idx), device=q.device)
            matrix_magnitude[:, spat_idx, range_idx] = float('inf') if self.magnitude_type == 'distance' else float('-inf')
            prime = self._prime_N(v, matrix_magnitude, self.K, spat_idx, self.maximum)
        else: 
            raise ValueError("Invalid sampling_type. Must be one of ['all', 'random', 'spatial']")

        x = self.conv(prime)  
        x = self.W_o(self.combine_heads(self.batch_split(x))).permute(0, 2, 1)
        return x

    def _calculate_similarity_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  
        return similarity_matrix
        
    def _calculate_distance_matrix(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True) 
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True) 
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        dist_matrix = norm_squared_K + norm_squared_Q.transpose(2, 1) - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0)  
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix
        return dist_matrix

    def _calculate_distance_matrix_N(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        dist_matrix = norm_squared_K + norm_squared_Q - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0)  
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix
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
    
class MultiHeadConv1dAttention(nn.Module):
    def __init__(self, d_hidden, num_heads, kernel_size): 
        super(MultiHeadConv1dAttention, self).__init__()
    
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_k = d_hidden // num_heads
        
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = (self.kernel_size - 1) // 2
        
        self.W_x = nn.Linear(d_hidden, d_hidden)
        self.W_o = nn.Linear(d_hidden, d_hidden)

        self.in_channels = d_hidden // num_heads
        self.out_channels = d_hidden // num_heads
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding, 
        )
        
    def split_head(self, x): 
        batch_size, seq_length, d_hidden = x.size()
        self.batch_size = batch_size
        self.seq_length = seq_length
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden) 
    
    def batch_split(self, x): 
        x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
        return x.permute(0, 1, 3, 2).contiguous()
        
    def batch_combine(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)       
    
    def forward(self, x):
        x = self.batch_combine(self.split_head(self.W_x(x)))
        x = self.conv(x) 
        x = self.W_o(self.combine_heads(self.batch_split(x.permute(0, 2, 1))))
        return x
        
class MultiHeadConv1d(nn.Module):
    def __init__(self, d_hidden, num_heads, kernel_size, seq_length=197):
        super(MultiHeadConv1d, self).__init__()
        
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_k = d_hidden // num_heads
        
        self.kernel_size = kernel_size
        self.stride = 1 
        self.padding = (self.kernel_size - 1) // 2
        self.seq_length = seq_length
        
        self.W_x = nn.Linear(self.seq_length, self.seq_length)
        self.W_o = nn.Linear(self.seq_length, self.seq_length)
        
        self.in_channels = d_hidden // num_heads
        self.out_channels = d_hidden // num_heads
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding, 
        )
    
    def split_head(self, x):
        batch_size, d_hidden, seq_length = x.size()
        self.batch_size = batch_size
        self.seq_length = seq_length
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
    
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, self.d_hidden, seq_length) 
    
    def batch_split(self, x): 
        x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
        return x.permute(0, 1, 3, 2).contiguous()
        
    def batch_combine(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)

    def forward(self, x):    
        x = x.permute(0, 2, 1)  # Change shape to (B, seq_length, d_hidden)
        x = self.batch_combine(self.split_head(self.W_x(x)))
        x = self.conv(x) 
        x = self.W_o(self.combine_heads(self.batch_split(x))).permute(0, 2, 1)
        return x

class MultiHeadKvtAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,topk=100):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.topk = topk

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # the core code block
        mask=torch.zeros(B,self.num_heads,N,N,device=x.device,requires_grad=False)
        index=torch.topk(attn,k=self.topk,dim=-1,largest=True)[1]
        mask.scatter_(-1,index,1.)
        attn=torch.where(mask>0,attn,torch.full_like(attn,float('-inf')))
        # end of the core code block

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
if __name__ == "__main__":
    import torch
    from types import SimpleNamespace
    
    # ViT-Small configuration
    args = SimpleNamespace(
        img_size = (3, 224, 224),       # (channels, height, width)
        patch_size = 16,                # 16x16 patches
        num_layers = 3,                 # 8 transformer layers
        num_heads = 4,                  # 8 attention heads
        d_hidden = 8,                 # Hidden dimension
        d_mlp = 24,                   # MLP dimension
        num_classes = 100,              # CIFAR-100 classes
        K = 9,                          # For nearest neighbor operations
        kernel_size = 9,                # Kernel size for ConvNN
        dropout = 0.1,                  # Dropout rate
        attention_dropout = 0.1,        # Attention dropout
        sampling_type = "spatial",          # Sampling type: 'all', 'random', or 'spatial'
        sample_padding = 3,             # Padding for spatial sampling
        num_samples = 32,               # Number of samples for random or spatial sampling; -
        magnitude_type = "similarity",  # Or "distance"
        shuffle_pattern = "NA",         # Default pattern
        shuffle_scale = 1,              # Default scale
        layer = "Attention",            # Attention or ConvNN
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else "cpu"),
        model = "ViT"                   # Model type
    )
    
    # Create the model
    model = ViT(args)
    
    print("Regular Attention")
    # Print parameter count
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    x = torch.randn(64, 3, 224, 224).to(args.device)  
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}\n")
    
    
    print("ConvNN")
    args.layer = "ConvNN"
    model = ViT(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    output_convnn = model(x)
    
    print(f"Output shape: {output_convnn.shape}\n")
    
    
    print("ConvNNAttention")
    args.layer = "ConvNNAttention"
    model = ViT(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    output = model(x)
    
    print(f"Output shape: {output.shape}\n")
    

    print("Conv1d")
    args.layer = "Conv1d"
    model = ViT(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    output = model(x)
    
    print(f"Output shape: {output.shape}\n")
    
    
    print("Conv1dAttention")
    args.layer = "Conv1dAttention"
    model = ViT(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    output = model(x)
    
    print(f"Output shape: {output.shape}\n")


    print("KvtAttention")
    args.layer = "KvtAttention"
    model = ViT(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    output = model(x)
    
    print(f"Output shape: {output.shape}")
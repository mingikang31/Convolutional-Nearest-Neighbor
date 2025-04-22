
#### NEW IMPLEMNETATION THAT I MIGHT TRY BUT DISREGARD UNTIL THEN ###
'''Convolution 1D Nearest Neighbors Attention Layer'''

'''
Conv1d_NN_Attn is a variation of Conv1d_NN that incorporates the attention mechanism.
'''

import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D
import numpy as np


# ... (keep original Conv1d_NN_Attn class definition if needed) ...

'''Convolution 1D Nearest Neighbors Attention Layer'''

'''
Conv1d_NN_Attn is a variation of Conv1d_NN that incorporates the attention mechanism.
'''

import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
# Make sure pixelshuffle is importable, e.g., from a local file or installed package
# from pixelshuffle import PixelShuffle1D, PixelUnshuffle1D
# Placeholder if pixelshuffle is not available:
class PixelShuffle1D(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
    def forward(self, x):
        b, c, t = x.shape
        out_c = c // self.upscale_factor
        out_t = t * self.upscale_factor
        x = x.view(b, out_c, self.upscale_factor, t)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(b, out_c, out_t)
        return x

class PixelUnshuffle1D(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor
    def forward(self, x):
        b, c, t = x.shape
        in_c = c * self.downscale_factor
        in_t = t // self.downscale_factor
        x = x.view(b, c, in_t, self.downscale_factor)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(b, in_c, in_t)
        return x
# End Placeholder
import numpy as np


class Conv1d_NN_Attn_v2(nn.Module):
    """
    Convolutional 1D Nearest Neighbors Attention Layer (Version 2)
    Applies QKV to channel dimension and uses 1x1 Conv on gathered features.
    """

    def __init__(self,
                in_channels,
                out_channels,
                K=3,
                stride=1, # Default stride to 1 for 1x1 conv usually
                padding=0,
                shuffle_pattern='N/A',
                shuffle_scale=2,
                samples='all',
                magnitude_type='similarity',
                # num_tokens parameter removed as QKV now operate on channels
                ):

        """
        Initializes the Conv1d_NN_Attn_v2 module.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration (includes self).
            stride (int): Stride size for the final 1x1 convolution.
            padding (int): Padding size for the final 1x1 convolution.
            shuffle_pattern (str): Shuffle pattern: "N/A", "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            samples (int/str): Number of samples to consider for KNN ('all' or int).
            magnitude_type (str): 'distance' or 'similarity'.
        """
        # --- Correction: Call super for the correct class ---
        super(Conv1d_NN_Attn_v2, self).__init__()

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

        # Determine channels before/after shuffle/unshuffle
        # C_in: Channels after potential unshuffle
        C_in = in_channels * self.shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        # C_out_conv: Channels expected by the shuffle layer (if used) or final output channels
        C_out_conv = out_channels * self.shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels

        # --- Modified Conv1d Layer ---
        # Processes concatenated neighbor features (C_in * K) for each token position.
        self.conv1d_layer = Conv1d(in_channels=C_in * self.K,
                                    out_channels=C_out_conv,
                                    kernel_size=1, # Use 1x1 convolution
                                    stride=self.stride, # Apply original stride here
                                    padding=self.padding) # Padding might need adjustment

        # --- Modified Linear Layers for Query, Key, Value ---
        # Operate on the channel dimension (C_in)
        self.q = nn.Linear(C_in, C_in, bias=False)
        self.k = nn.Linear(C_in, C_in, bias=False)
        self.v = nn.Linear(C_in, C_in, bias=False)


    def forward(self, x):
        # Input shape: [B, C_orig, T_orig]

        # Unshuffle Layer
        if self.shuffle_pattern in ["B", "BA"]:
            x1 = self.unshuffle_layer(x) # Shape: [B, C_in, T]
        else:
            x1 = x # Shape: [B, C_in, T]

        b, c_in, t = x1.shape

        # --- Apply Q, K, V to Channel Dimension ---
        # Permute to [B, T, C_in] for nn.Linear which acts on the last dim
        x1_transposed = x1.permute(0, 2, 1) # Shape: [B, T, C_in]
        q_transposed = self.q(x1_transposed) # Shape: [B, T, C_in]
        k_transposed = self.k(x1_transposed) # Shape: [B, T, C_in]
        v_transposed = self.v(x1_transposed) # Shape: [B, T, C_in]

        # Permute back to [B, C_in, T] for similarity/distance calculation
        q = q_transposed.permute(0, 2, 1) # Shape: [B, C_in, T]
        k = k_transposed.permute(0, 2, 1) # Shape: [B, C_in, T]
        v = v_transposed.permute(0, 2, 1) # Shape: [B, C_in, T]


        # Consider all samples
        if self.samples == 'all':
            # Calculate Distance/Similarity Matrix
            if self.magnitude_type == 'distance':
                # k, q shapes: [B, C_in, T]
                matrix_magnitude = self._calculate_distance_matrix(k, q, sqrt=True) # Shape: [B, T, T]
            elif self.magnitude_type == 'similarity':
                # k, q shapes: [B, C_in, T]
                matrix_magnitude = self._calculate_similarity_matrix(k, q) # Shape: [B, T, T]

            # Gather neighbors based on magnitude
            # v shape: [B, C_in, T], matrix_magnitude shape: [B, T, T]
            prime = self._prime(v, matrix_magnitude, self.K, self.maximum) # Shape: [B, C_in, T*K]

        # Consider N samples
        else:
            # Calculate Distance/Similarity Matrix + Prime
            # Ensure self.samples is not larger than sequence length t
            num_samples = min(self.samples, t)
            if num_samples < self.K -1:
                 print(f"Warning: Number of samples ({num_samples}) is less than K-1 ({self.K-1}). Consider increasing samples or decreasing K.")
                 # Handle this case: maybe use all samples or raise error
                 num_samples = t # Fallback to using all samples if too few

            rand_idx = torch.randperm(t, device=x1.device)[:num_samples]
            k_sample = k[:, :, rand_idx] # Shape: [B, C_in, M] where M = num_samples
            q_query = q # Use all queries: Shape [B, C_in, T]

            if self.magnitude_type == 'distance':
                # k_sample: [B, C_in, M], q_query: [B, C_in, T]
                matrix_magnitude = self._calculate_distance_matrix_N(k_sample, q_query, sqrt=True) # Shape: [B, M, T]
            elif self.magnitude_type == 'similarity':
                 # k_sample: [B, C_in, M], q_query: [B, C_in, T]
                matrix_magnitude = self._calculate_similarity_matrix_N(k_sample, q_query) # Shape: [B, M, T]

            # Transpose magnitude matrix to [B, T, M] for topk and prime_N logic
            matrix_magnitude = matrix_magnitude.transpose(1, 2) # Shape: [B, T, M]

            # Mask self-comparison is tricky here as we compare all T queries to M keys
            # The _prime_N function handles adding the self-token back

            # Gather neighbors based on magnitude (includes self via _prime_N)
            # v shape: [B, C_in, T], matrix_magnitude shape: [B, T, M]
            prime = self._prime_N(v, matrix_magnitude, self.K, rand_idx, self.maximum) # Shape: [B, C_in, T*K]


        # --- Reshape Prime Tensor ---
        # Reshape prime: [B, C_in, T*K] -> [B, C_in, T, K] -> [B, C_in*K, T]
        b_prime, c_prime, tk_prime = prime.shape
        # Ensure T*K is divisible by K
        if tk_prime % self.K != 0:
            raise ValueError(f"Shape mismatch: prime tensor dim 2 ({tk_prime}) not divisible by K ({self.K})")
        t_prime = tk_prime // self.K # This should be equal to t
        if t_prime != t:
             print(f"Warning: Inferred sequence length {t_prime} from prime tensor differs from actual {t}")
             # This might indicate an issue in _prime or _prime_N if t_prime != t

        # Reshape and permute
        prime_reshaped = prime.view(b_prime, c_prime, t_prime, self.K) # [B, C_in, T, K]
        prime_reshaped = prime_reshaped.permute(0, 1, 3, 2).contiguous() # [B, C_in, K, T]
        prime_reshaped = prime_reshaped.view(b_prime, c_prime * self.K, t_prime) # [B, C_in*K, T]


        # --- Apply Modified Conv1d Layer ---
        x2 = self.conv1d_layer(prime_reshaped) # Shape: [B, C_out_conv, T']


        # Shuffle Layer
        if self.shuffle_pattern in ["A", "BA"]:
            x3 = self.shuffle_layer(x2) # Shape: [B, C_out, T''']
        else:
            x3 = x2 # Shape: [B, C_out, T']

        return x3


    def _calculate_similarity_matrix(self, K_in, Q_in):
        # K_in, Q_in shape: [B, C, N], [B, C, M] (Here N=M=T)
        k_norm = F.normalize(K_in, p=2, dim=1)
        q_norm = F.normalize(Q_in, p=2, dim=1)
        # k_norm.transpose(1, 2) shape: [B, N, C]
        # q_norm shape: [B, C, M]
        similarity_matrix = torch.bmm(k_norm.transpose(1, 2), q_norm)  # Shape: [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)
        return similarity_matrix # Shape: [B, T, T]

    def _calculate_similarity_matrix_N(self, K_in, Q_in):
        # K_in shape: [B, C, N] (N=num_samples=M)
        # Q_in shape: [B, C, M] (M=num_queries=T)
        k_norm = F.normalize(K_in, p=2, dim=1)
        q_norm = F.normalize(Q_in, p=2, dim=1)
        # k_norm.transpose(1, 2) shape: [B, N, C]
        # q_norm shape: [B, C, M]
        similarity_matrix = torch.bmm(k_norm.transpose(1, 2), q_norm)  # Shape: [B, N, M]
        similarity_matrix = torch.clamp(similarity_matrix, min=0)
        return similarity_matrix # Shape: [B, M, T]


    def _calculate_distance_matrix(self, K_in, Q_in, sqrt=False):
        # K_in, Q_in shape: [B, C, N], [B, C, M] (Here N=M=T)
        norm_squared_K = torch.sum(K_in**2, dim=1, keepdim=True) # Shape: [B, 1, N]
        norm_squared_Q = torch.sum(Q_in**2, dim=1, keepdim=True) # Shape: [B, 1, M]

        # K_in.transpose(1, 2) shape: [B, N, C]
        # Q_in shape: [B, C, M]
        dot_product = torch.bmm(K_in.transpose(1, 2), Q_in) # Shape: [B, N, M]

        # Broadcasting: [B, N, 1] + [B, 1, M] - 2*[B, N, M] -> [B, N, M]
        dist_matrix = norm_squared_K.transpose(1,2) + norm_squared_Q - 2 * dot_product

        dist_matrix = torch.clamp(dist_matrix, min=0)

        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)

        return dist_matrix # Shape: [B, T, T]

    def _calculate_distance_matrix_N(self, K_in, Q_in, sqrt=False):
         # K_in shape: [B, C, N] (N=num_samples=M)
         # Q_in shape: [B, C, M] (M=num_queries=T)
        norm_squared_K = torch.sum(K_in**2, dim=1, keepdim=True) # Shape: [B, 1, N]
        norm_squared_Q = torch.sum(Q_in**2, dim=1, keepdim=True) # Shape: [B, 1, M]

        # K_in.transpose(1, 2) shape: [B, N, C]
        # Q_in shape: [B, C, M]
        dot_product = torch.bmm(K_in.transpose(1, 2), Q_in) # Shape: [B, N, M]

        # Broadcasting: [B, N, 1] + [B, 1, M] - 2*[B, N, M] -> [B, N, M]
        dist_matrix = norm_squared_K.transpose(1,2) + norm_squared_Q - 2 * dot_product

        dist_matrix = torch.clamp(dist_matrix, min=0)

        if sqrt:
            dist_matrix = torch.sqrt(dist_matrix)

        return dist_matrix # Shape: [B, M, T]

# ... existing code ...

    def _prime(self, v, qk, K, maximum):
        # v shape: [B, C, T]
        # qk shape: [B, T, T] (N=T, M=T)
        b, c, t = v.shape

        # Find top K indices along the key dimension (dim=-1 or dim=2) of qk
        # For each query token (dim=1), find the K closest key tokens (dim=2)
        _, topk_indices = torch.topk(qk, k=K, dim=-1, largest = maximum) # Shape: [B, T, K]

        # Expand indices for gathering across the channel dimension.
        # Shape needs to match v's dims B and C, and have T and K for indexing.
        # Target index shape for gather(dim=2): [B, C, T, K]
        indices_expanded = topk_indices.unsqueeze(1).expand(b, c, t, K) # Shape: [B, C, T, K]

        # --- Correction Start ---
        # Gather requires index.ndim == input.ndim.
        # We need to gather from v [B, C, T] using indices [B, C, T, K] along dim 2.
        # Expand v to match the non-gathering dimensions of the index tensor.
        # Expand v: [B, C, T] -> [B, C, T, 1] -> [B, C, T, K]
        # This allows selecting T*K elements based on the indices.
        v_expanded_for_gather = v.unsqueeze(3).expand(b, c, t, K) # Shape: [B, C, T, K]

        # Gather along the token dimension (dim=2)
        # output[b, c, t_query, k_idx] = input[b, c, index[b, c, t_query, k_idx], k_idx]
        # Here, input is v_expanded_for_gather, index is indices_expanded
        # output[b,c,t,k] = v_expanded_for_gather[b, c, indices_expanded[b,c,t,k], k]
        # Since v_expanded_for_gather[b,c,i,k] = v[b,c,i], this becomes:
        # output[b,c,t,k] = v[b, c, indices_expanded[b,c,t,k]]
        # which is the desired operation: selecting the k-th neighbor's value for token t.
        prime_gathered = torch.gather(v_expanded_for_gather, dim=2, index=indices_expanded) # Shape: [B, C, T, K]
        # --- Correction End ---


        # Reshape to [B, C, T*K]
        prime = prime_gathered.reshape(b, c, -1)

        return prime

    def _prime_N(self, v, qk, K, rand_idx, maximum):
        # v shape: [B, C, T]
        # qk shape: [B, T, M] (M = num_samples)
        # rand_idx: [M] (indices of the keys used in qk)
        b, c, t = v.shape
        m = qk.shape[2]

        # For each query token (dim=1), find top-(K-1) indices from the M sampled key tokens (dim=2)
        k_neighbors_to_find = K - 1
        if k_neighbors_to_find > m:
            print(f"Warning: K-1 ({k_neighbors_to_find}) > num_samples ({m}). Finding only {m} neighbors.")
            k_neighbors_to_find = m

        if k_neighbors_to_find < 0: # Handle K=0 case if necessary, though K>=1 usually
             k_neighbors_to_find = 0

        if K == 1: # Special case K=1, only self-token
             token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1) # [B, T, 1]
             final_indices = token_indices
             final_k = 1
        elif k_neighbors_to_find == 0: # K > 1 but samples are 0 or K-1 <= 0
             print(f"Warning: K={K}, but k_neighbors_to_find is 0. Only using self-token.")
             token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1) # [B, T, 1]
             final_indices = token_indices
             final_k = 1
        else:
            _, topk_indices_sampled = torch.topk(qk, k=k_neighbors_to_find, dim=2, largest=maximum) # Shape: [B, T, k_neighbors_to_find]
            actual_neighbors_found = topk_indices_sampled.shape[-1]

            # Map sampled indices back to original token indices in the full sequence T
            rand_idx_expanded = rand_idx.view(1, 1, m).expand(b, t, m)
            mapped_indices = torch.gather(rand_idx_expanded, dim=2, index=topk_indices_sampled) # Shape: [B, T, actual_neighbors_found]

            # Create self indices for each token
            token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1) # Shape: [B, T, 1]

            # Concatenate self index with neighbor indices
            final_indices = torch.cat([token_indices, mapped_indices], dim=2) # Shape: [B, T, 1 + actual_neighbors_found]
            final_k = final_indices.shape[-1] # This should ideally be K, but might be less

        # Expand indices for gathering: [B, T, final_k] -> [B, 1, T, final_k] -> [B, C, T, final_k]
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, final_k)

        # --- Correction Start (similar to _prime) ---
        # Expand v to match the non-gathering dimensions of the index tensor.
        # Expand v: [B, C, T] -> [B, C, T, 1] -> [B, C, T, final_k]
        v_expanded_for_gather = v.unsqueeze(3).expand(b, c, t, final_k) # Shape: [B, C, T, final_k]

        # Gather from v along the token dimension (dim=2)
        prime_gathered = torch.gather(v_expanded_for_gather, dim=2, index=indices_expanded) # Shape: [B, C, T, final_k]
        # --- Correction End ---

        # Reshape to [B, C, T*final_k]
        prime = prime_gathered.reshape(b, c, -1) # Shape [B, C, T*final_k]

        # Pad if fewer than K neighbors were found/used
        if final_k < K:
            padding_size = t * (K - final_k)
            prime = F.pad(prime, (0, padding_size)) # Pad on the right of the last dimension

        return prime # Shape [B, C, T*K]


# --- Updated example_usage ---
def example_usage():
    print("--- Running Example Usage for Conv1d_NN_Attn_v2 ---")
    ex = torch.randn(4, 3, 64) # Example: B=4, C=3, T=64

    print("\nCase 1: samples='all', shuffle='BA'")
    conv1d_NN_attn = Conv1d_NN_Attn_v2(in_channels=3,
                                    out_channels=6,
                                    K=5,
                                    stride=1, # Stride for the 1x1 conv
                                    padding=0,
                                    shuffle_pattern='BA', # Unshuffle->Attn->Conv->Shuffle
                                    shuffle_scale=2,
                                    samples='all',
                                    magnitude_type='similarity'
                                    )
    # Input: [4, 3, 64]
    # Unshuffle (scale=2): [4, 3*2, 64/2] = [4, 6, 32] (C_in=6, T=32)
    # Q,K,V linear: C_in=6 -> C_in=6
    # Mag matrix: [4, 32, 32]
    # Prime: [4, 6, 32*5] = [4, 6, 160]
    # Reshape prime: [4, 6*5, 32] = [4, 30, 32]
    # Conv1d (in=30, out=C_out_conv, k=1, s=1):
    #   C_out_conv = out_channels(6) * shuffle_scale(2) = 12
    #   Output shape: [4, 12, 32] (T'=T if s=1, p=0)
    # Shuffle (scale=2): [4, 12/2, 32*2] = [4, 6, 64]
    out_N = conv1d_NN_attn(ex)
    print(f'Input shape: {ex.shape}')
    print(f'Output shape (N samples, stride=2): {out_N.shape}') # Expected: [4, 10, 32]

    print("\nCase 3: K=1 (only self-token)")
    conv1d_NN_attn_K1 = Conv1d_NN_Attn_v2(in_channels=3,
                                    out_channels=6,
                                    K=1,
                                    stride=1,
                                    shuffle_pattern='N/A',
                                    samples='all',
                                    magnitude_type='similarity'
                                    )
    out_K1 = conv1d_NN_attn_K1(ex)
    print(f'Input shape: {ex.shape}')
    print(f'Output shape (K=1): {out_K1.shape}') # Expected: [4, 6, 64]


# --- Run the example ---
example_usage()
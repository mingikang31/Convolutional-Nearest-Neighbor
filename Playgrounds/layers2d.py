import torch 
import torch.nn as nn 
import torch.nn.functional as F
import time 
import math

# class Conv2d_NN_sanity(nn.Module):
#     def __init__(self, 
#             in_channels, 
#             out_channels, 
#             K,
#             stride, 
#             padding, 
#             sampling_type, 
#             num_samples, 
#             sample_padding, # NOT IN USE AS OF NOW
#             shuffle_pattern, 
#             shuffle_scale, 
#             magnitude_type,
#             coordinate_encoding
#                 ):
#         super(Conv2d_NN_sanity, self).__init__()

#         assert K == stride, "K must be equal to stride for ConvNN."


#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.K = K
#         self.stride = stride
#         self.padding = padding
#         self.sampling_type = sampling_type
#         self.num_samples = num_samples
#         self.sample_padding = sample_padding
#         self.shuffle_pattern = shuffle_pattern
#         self.shuffle_scale = shuffle_scale
#         self.magnitude_type = magnitude_type
#         self.coordinate_encoding = coordinate_encoding

#         # Shuffle2D/Unshuffle2D Layers 
#         self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
#         self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
#         # Positional Encoding (optional)
#         self.coordinate_encoding = coordinate_encoding
#         self.coordinate_cache = {} 
        
#         # Conv1d Layer 
#         self.conv1d_layer = nn.Conv1d(
#             in_channels=self.in_channels, # + 2, ## CHANGE IF NEEDED
#             out_channels=self.out_channels,
#             kernel_size=self.K,
#             stride=self.stride,
#         )

#         self.flatten = nn.Flatten(start_dim=2) 
#         self.unflatten = None


#         # Shapes of tensors
#         self.og_shape = None 
#         self.pad_shape = None

#         init_h, init_w = None, None 
#         padded_h, padded_w = None, None

#     def forward(self, x):
#         if not self.og_shape:
#             self.og_shape = x.shape
#         print("Original x shape: ", self.og_shape)
#         x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0) if self.padding > 0 else x
        
#         if not self.pad_shape:
#             self.pad_shape = x.shape
#         print("Padded x shape: ", self.pad_shape)

#         x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
#         print("coor shape: ", x.shape)
#         x = self.flatten(x)
#         print("flattened shape: ", x.shape)

#         x_dist = x[:, -2:, :]
#         x = x[:, :-2, :] 

#         if self.sampling_type == "all":
#             similarity_matrix = self._calculate_euclidean_matrix(x_dist)
#             prime = self._prime(x, similarity_matrix, self.K, maximum=True)
#         print("prime shape: ", prime.shape)
#         x = self.conv1d_layer(prime)
#         print("conv1d shape: ", x.shape)
#         # print(x.shape)
#         if not self.unflatten:
#             self.unflatten = nn.Unflatten(dim=2, unflattened_size=self.og_shape[2:])

#         x = self.unflatten(x)
#         print("unflattened shape: ", x.shape)
#         # print(x.shape)

#         print("final shape: ", x.shape)

#         print("sleeping for 2 seconds")
#         time.sleep(2)
#         return x


#     def _calculate_similarity_matrix(self, matrix, sigma=0.1):
#         """Calculate similarity matrix based on coordinate distance"""
#         b, c, t = matrix.shape  # c should be 2 for (x, y) coordinates

#         ### TODO CHANGE IF NOT USING DISTANCE ANYMORE
#         # coord_matrix = matrix[:, -2:, :]
#         coord_matrix = matrix

#         # Calculate pairwise Euclidean distances between coordinates
#         coord_expanded_1 = coord_matrix.unsqueeze(3)  # [B, 2, T, 1]
#         coord_expanded_2 = coord_matrix.unsqueeze(2)  # [B, 2, 1, T]

#         # Euclidean distance between coordinates
#         coord_diff = coord_expanded_1 - coord_expanded_2  # [B, 2, T, T]
#         coord_dist = torch.sqrt(torch.sum(coord_diff ** 2, dim=1) + 1e-8)  # [B, T, T]
        
#         # Convert distance to similarity using Gaussian kernel
#         similarity_matrix = torch.exp(-coord_dist ** 2 / (2 * sigma ** 2))

#         return similarity_matrix
    
#     def _calculate_euclidean_matrix(self, matrix, sqrt=False):
#         norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
#         dot_product = torch.bmm(matrix.transpose(2, 1), matrix)
#         dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
#         dist_matrix = torch.clamp(dist_matrix, min=0.0) 
#         dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix 
        
#         return dist_matrix
    
#     def _prime(self, matrix, magnitude_matrix, K, maximum):
#         b, c, t = matrix.shape
#         """ ORIGINAL
#         _, topk_indices = torch.topk(magnitude_matrix.detach(), k=K, dim=2, largest=maximum)
#         print("Top-k Indices")
#         print(topk_indices.shape)
#         """
#         # New My TopK
#         _, sorted_indices = torch.sort(magnitude_matrix.detach(), dim=2, descending=False, stable=True)
#         topk_indices = sorted_indices[:, :, :K]

#         # _, topk_indices = torch.topk(magnitude_matrix.detach(), k=K, dim=2, largest=False)
    
#         # End of My TopK
        
#         topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
#         matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
#         prime = torch.gather(matrix_expanded, dim=2, index=topk_indices_exp)
#         # prime, _ = self.filter_non_zero_starting_rows_multichannel(prime)
#         # b, c, num_filtered_rows, k = prime.shape
#         print()
#         print("With Padding Ks")
#         print(prime.shape)
#         print(prime)
#         print()
#         if self.padding > 0:
#             prime = prime.view(b, c, self.pad_shape[-2], self.pad_shape[-1], K)
#             print("Prime with Padded shape:")
#             print(prime.shape)
#             print(prime)
#             print()
#             prime = prime[:, :, self.padding:-self.padding, self.padding:-self.padding, :]
#             print("Without Padding Ks")
#             print(prime.shape)
#             print(prime)

#             prime = prime.reshape(b, c, K * self.og_shape[-2] * self.og_shape[-1])
#         else: 
#             prime = prime.view(b, c, -1)
        
#         return prime

#     def sort_indices_within_tied_values(self, sorted_indices, sorted_values):
#         """
#         Sort indices in ascending order when values are tied.
        
#         Args:
#             sorted_indices: tensor of shape [batch, points, k]
#             sorted_values: tensor of shape [batch, points, k]
        
#         Returns:
#             Tuple of (new_sorted_indices, new_sorted_values)
#         """
#         # Create a compound sorting key: primary sort by value, secondary by index
#         # Scale indices to be much smaller than value differences
#         index_scale = 1e-10  # Small enough to not affect value ordering
#         compound_key = sorted_values + sorted_indices.float() * index_scale
        
#         # Sort by compound key
#         _, reorder_indices = torch.sort(compound_key, dim=2)
        
#         # Reorder both arrays using the new ordering
#         new_sorted_indices = torch.gather(sorted_indices, 2, reorder_indices)
#         new_sorted_values = torch.gather(sorted_values, 2, reorder_indices)
        
#         return new_sorted_indices, new_sorted_values

#     def filter_non_zero_starting_rows_multichannel(self, tensor):
#         """
#         Filter rows based on the first element of the first channel being non-zero
        
#         Args:
#             tensor: Input tensor of shape [B, C, num_rows, row_length]
        
#         Returns:
#             Filtered tensor with only rows where first channel's first element != 0
#         """
#         # Get the shape
#         b, c, num_rows, row_length = tensor.shape
        
#         # Create mask based on first channel only
#         # tensor[:, 0, :, 0] gets first element of each row in first channel
            
#         mask = tensor[:, 0, :, 0].detach() != 0  # Shape: [b, num_rows]
        
#         # Get indices of non-zero starting rows
#         non_zero_indices = torch.where(mask[0])[0]  # [0] because batch dimension
    
#         # Select rows from ALL channels
#         filtered_tensor = tensor[:, :, non_zero_indices, :]
        
#         return filtered_tensor, non_zero_indices

#     def _add_coordinate_encoding(self, x):
#         b, _, h, w = x.shape
#         cache_key = f"{b}_{h}_{w}_{x.device}"

#         if cache_key in self.coordinate_cache:
#             expanded_grid = self.coordinate_cache[cache_key]
#         else:
#             with torch.no_grad():
#                 y_coords_vec = torch.linspace(start=-1, end=1, steps=h, device=x.device)
#                 x_coords_vec = torch.linspace(start=-1, end=1, steps=w, device=x.device)

#                 y_grid, x_grid = torch.meshgrid(y_coords_vec, x_coords_vec, indexing='ij')
#                 grid = torch.stack((x_grid, y_grid), dim=0).unsqueeze(0)
#                 expanded_grid = grid.expand(b, -1, -1, -1)
#                 self.coordinate_cache[cache_key] = expanded_grid

#         x_with_coords = torch.cat((x, expanded_grid), dim=1)
#         return x_with_coords 



class Conv2d_NN_sanity(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 K, 
                 stride, 
                 padding, 
                 sampling_type, 
                 num_samples, 
                 sample_padding,
                 shuffle_pattern, 
                 shuffle_scale, 
                 magnitude_type, 
                 similarity_type, 
                 aggregation_type
                ):

        super(Conv2d_NN_sanity, self).__init__()

        assert K == stride, "K must be equal to stride for ConvNN"
        assert padding > 0 or padding == 0, "Cannot have Negative Padding"
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Shuffle pattern must be: Before, After, Before After, Not Applicable"
        assert magnitude_type in ["cosine", "euclidean"], "Similarity Matrix must be either cosine similarity or euclidean distance"
        assert sampling_type in ["all", "random", "spatial"], "Consider all neighbors, random neighbors, or spatial neighbors"
        assert int(num_samples) > 0 or int(num_samples) == -1, "Number of samples to consider must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and int(num_samples) == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Number of samples must be -1 for all samples or integer for random and spatial sampling"

        assert similarity_type in ["Loc", "Col", "Loc_Col"], "Similarity Matrix based on Location, Color, or both"
        assert aggregation_type in ["Col", "Loc_Col"], "Aggregation based on Color or Location and Color"

        # Core Parameters
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.K = K
        self.stride = stride 
        self.padding = padding 

        # 3 Sampling Types: all, random, spatial
        self.sampling_type = sampling_type
        self.num_samples = num_samples
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0

        # Pixel Shuffling (optional) 
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale

        # Similarity Metric
        self.magnitude_type = magnitude_type
        self.maximum = True if magnitude_type == "cosine" else False

        # Similarity and Aggregation Types
        self.similarity_type = similarity_type
        self.aggregation_type = aggregation_type
        
        # Positional Encoding (optional)
        self.coordinate_encoding = True if (similarity_type in ["Loc", "Loc_Col"] or aggregation_type == "Loc_Col") else False
        self.coordinate_cache = {}

        # Pixel Shuffle Adjustments
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale) 
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)

        self.in_channels_1d = self.in_channels * (self.shuffle_scale ** 2) if self.shuffle_pattern in ["B", "BA"] else self.in_channels
        self.out_channels_1d = self.out_channels * (self.shuffle_scale ** 2) if self.shuffle_pattern in ["A", "BA"] else self.out_channels

        self.in_channels_1d = self.in_channels_1d + 2 if self.aggregation_type == "Loc_Col" else self.in_channels_1d

        self.conv1d_layer = nn.Conv1d(
            in_channels = self.in_channels_1d,
            out_channels = self.out_channels_1d,
            kernel_size = self.K, 
            stride = self.stride, 
            padding = 0, 
            bias = False
        )

        # Flatten * Unflatten layers 
        self.flatten = nn.Flatten(start_dim=2)
        self.unflatten = None

        self.og_shape = None 
        self.padded_shape = None


        # Utility Variables
        self.INF = 1e5
        self.NEG_INF = -1e5


    def forward(self, x):  
        # 1. Pixel Shuffle 
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        self.og_shape = x.shape

        # 2. Add Padding 
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
            self.padded_shape = x.shape

        # 3. Add Coordinate Encoding
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x

        # 4. Flatten 
        x = self.flatten(x) 

        # 5. Similarity and Aggregation Type 
        if self.similarity_type == "Loc":
            x_sim = x[:, -2:, :]
        elif self.similarity_type == "Loc_Col":
            x_sim = x
        elif self.similarity_type == "Col" and self.aggregation_type == "Col":
            x_sim = x
        elif self.similarity_type == "Col" and self.aggregation_type == "Loc_Col":
            x_sim = x[:, :-2, :]

        if self.similarity_type in ["Loc", "Loc_Col"] and self.aggregation_type == "Col":
            x = x[:, :-2, :]
        else: 
            x = x
            
        # if self.similarity_type == "Loc_Col":
        #     x_sim = x_sim/math.sqrt(2)
        #     x = x/math.sqrt(self.og_shape[1])
            
        # 4. Sampling + Similarity Calculation + Aggregation
        if self.sampling_type == "all":
            similarity_matrix = self._calculate_euclidean_matrix(x_sim) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix(x_sim)
            prime = self._prime(x, similarity_matrix, self.K, self.maximum)

        elif self.sampling_type == "random":
            rand_idx = torch.randperm(x.shape[-1], device=x.device)[:self.num_samples]
            x_sample = x_sim[:, :, rand_idx]

            similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample, sqrt=True) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(x_sim, x_sample)

            range_idx = torch.arange(len(rand_idx), device=x.device)
            similarity_matrix[:, rand_idx, range_idx] = self.INF if self.magnitude_type == "euclidean" else self.NEG_INF

            prime = self._prime_N(x, similarity_matrix, self.K, rand_idx, self.maximum)

        elif self.sampling_type == "spatial":
            x_ind = torch.linspace(0 + self.sample_padding, self.og_shape[-2] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            y_ind = torch.linspace(0 + self.sample_padding, self.og_shape[-1] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
            x_idx_flat, y_idx_flat = x_grid.flatten(), y_grid.flatten()
            width = self.og_shape[-2]
            flat_indices = y_idx_flat * width + x_idx_flat
            x_sample = x_sim[:, :, flat_indices]

            similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample, sqrt=True) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(x_sim, x_sample)

            range_idx = torch.arange(len(flat_indices), device=x.device)    
            similarity_matrix[:, flat_indices, range_idx] = self.INF if self.magnitude_type == "euclidean" else self.NEG_INF
            prime = self._prime_N(x, similarity_matrix, self.K, flat_indices, self.maximum)
        else:
            raise NotImplementedError("Sampling Type not Implemented")
        
        # 5. Conv1d Layer
        x = self.conv1d_layer(prime)

        if not self.unflatten: 
            self.unflatten = nn.Unflatten(dim=2, unflattened_size=self.og_shape[2:])
        x = self.unflatten(x)
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x 
        return x 

    def _calculate_euclidean_matrix(self, matrix, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dot_product = torch.matmul(matrix.transpose(2, 1), matrix)
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix 
        dist_matrix = torch.clamp(dist_matrix, min=0.0) 

        torch.diagonal(dist_matrix, dim1=1, dim2=2).fill_(-0.1)
        return dist_matrix
    
    def _calculate_euclidean_matrix_N(self, matrix, matrix_sample, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        norm_squared_sample = torch.sum(matrix_sample ** 2, dim=1, keepdim=True)
        dot_product = torch.bmm(matrix.transpose(1, 2), matrix_sample)
        
        dist_matrix = norm_squared.transpose(1, 2) + norm_squared_sample - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0.0) 
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix

        return dist_matrix
    
    def _calculate_cosine_matrix(self, matrix):
        # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        norm_matrix = F.normalize(matrix, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_matrix.transpose(2, 1), norm_matrix)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0) 
        torch.diagonal(similarity_matrix, dim1=1, dim2=2).fill_(1.1)
        return similarity_matrix
    
    def _calculate_cosine_matrix_N(self, matrix, matrix_sample):
        # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        norm_matrix = F.normalize(matrix, p=2, dim=1) 
        norm_sample = F.normalize(matrix_sample, p=2, dim=1)
        similarity_matrix = torch.bmm(norm_matrix.transpose(2, 1), norm_sample)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0) 
        return similarity_matrix
    
    def _prime(self, matrix, magnitude_matrix, K, maximum):
        b, c, t = matrix.shape

        if self.similarity_type == "Loc":
            _, topk_indices = torch.sort(magnitude_matrix, dim=2, descending=maximum, stable=True)
            topk_indices = topk_indices[:, :, :K]
        else:
            _, topk_indices = torch.topk(magnitude_matrix, k=K, dim=2, largest=maximum)

        # _new, topk_indices_new = self.sort_indices_within_tied_values(topk_indices, _)
        topk_indices, _ = torch.sort(topk_indices, dim=-1)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    

        # print("topk_indices shape:", topk_indices.shape)
        # print("topk_indices: ", topk_indices)
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=topk_indices_exp)

        if self.padding > 0: 
            prime = prime.view(b, c, self.padded_shape[-2], self.padded_shape[-1], K)
            prime = prime[:, :, self.padding:-self.padding, self.padding:-self.padding, :]
            prime = prime.reshape(b, c, K * self.og_shape[-2] * self.og_shape[-1])
        else: 
            prime = prime.view(b, c, -1)

        return prime
        
    def _prime_N(self, matrix, magnitude_matrix, K, rand_idx, maximum):
        b, c, t = matrix.shape
        
        _, topk_indices = torch.topk(magnitude_matrix, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # print("topk_indices shape:", topk_indices.shape)
        # print("topk_indices: ", topk_indices)

        
        # Map sample indices back to original matrix positions
        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=matrix.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
        final_indices, _ = torch.sort(final_indices, dim=-1)
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Gather matrix values and apply similarity weighting
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=indices_expanded)  

        if self.padding > 0:
            prime = prime.view(b, c, self.padded_shape[-2], self.padded_shape[-1], K)
            prime = prime[:, :, self.padding:-self.padding, self.padding:-self.padding, :]
            prime = prime.reshape(b, c, K * self.og_shape[-2] * self.og_shape[-1])
        else:
            prime = prime.view(b, c, -1)
        return prime

    def _add_coordinate_encoding(self, x):
        b, _, h, w = x.shape
        cache_key = f"{b}_{h}_{w}_{x.device}"

        if cache_key in self.coordinate_cache:
            expanded_grid = self.coordinate_cache[cache_key]
        else:
            y_coords_vec = torch.linspace(start=-1, end=1, steps=h, device=x.device)
            x_coords_vec = torch.linspace(start=-1, end=1, steps=w, device=x.device)

            y_grid, x_grid = torch.meshgrid(y_coords_vec, x_coords_vec, indexing='ij')
            grid = torch.stack((x_grid, y_grid), dim=0).unsqueeze(0)
            expanded_grid = grid.expand(b, -1, -1, -1)
            self.coordinate_cache[cache_key] = expanded_grid

        x_with_coords = torch.cat((x, expanded_grid), dim=1)
        return x_with_coords ### Last two channels are coordinate channels 
    
    def sort_indices_within_tied_values(self, sorted_values, sorted_indices):    # First sort by indices (secondary sort)
        idx_sort = torch.argsort(sorted_indices, dim=-1, stable=True)
        temp_indices = torch.gather(sorted_indices, -1, idx_sort)
        temp_values = torch.gather(sorted_values, -1, idx_sort)
        
        # Then sort by values (primary sort, stable preserves index order for ties)
        val_sort = torch.argsort(temp_values, dim=-1, stable=True)
        final_indices = torch.gather(temp_indices, -1, val_sort)
        final_values = torch.gather(temp_values, -1, val_sort)
        
        return final_indices, final_values
        
        return new_sorted_values, new_sorted_indices
if __name__ == "__main__":
    ex = torch.Tensor(
        [
            [
                [
                    [1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12],
                    [13, 14, 15, 16, 17, 18],
                    [19, 20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29, 30],
                    [31, 32, 33, 34, 35, 36]
                ], 
                [
                    [1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12],
                    [13, 14, 15, 16, 17, 18],
                    [19, 20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29, 30],
                    [31, 32, 33, 34, 35, 36]
                ], 
                [
                    [1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12],
                    [13, 14, 15, 16, 17, 18],
                    [19, 20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29, 30],
                    [31, 32, 33, 34, 35, 36]
                ]
            ]
        ]
    )

    print(ex)
    conv = Conv2d_NN_sanity(
        in_channels=3,
        out_channels=5,
        K=9,
        stride=9,
        padding=1,
        sampling_type="all",
        num_samples=-1,
        sample_padding=0,
        shuffle_pattern="NA",
        shuffle_scale=1.0,
        magnitude_type="euclidean",
        similarity_type = "Loc", aggregation_type = "Col" 
    )

    out = conv(ex)
    print(out.shape)





class Conv2d_NN_sanity_Dist(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            K,
            stride, 
            padding, 
            sampling_type, 
            num_samples, 
            sample_padding, # NOT IN USE AS OF NOW
            shuffle_pattern, 
            shuffle_scale, 
            magnitude_type,
            coordinate_encoding
                ):
        super(Conv2d_NN_sanity, self).__init__()

        assert K == stride, "K must be equal to stride for ConvNN."


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding
        self.sampling_type = sampling_type
        self.num_samples = num_samples
        self.sample_padding = sample_padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.magnitude_type = magnitude_type
        self.coordinate_encoding = coordinate_encoding

        # Shuffle2D/Unshuffle2D Layers 
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Positional Encoding (optional)
        self.coordinate_encoding = coordinate_encoding
        self.coordinate_cache = {} 
        
        # Conv1d Layer 
        self.conv1d_layer = nn.Conv1d(
            in_channels=self.in_channels, # + 2 if self.coordinate_encoding else self.in_channels, 
            out_channels=self.out_channels,
            kernel_size=self.K,
            stride=self.stride,
        )

        self.flatten = nn.Flatten(start_dim=2) 
        self.unflatten = None


        # Shapes of tensors
        self.og_shape = None 
        self.pad_shape = None

        init_h, init_w = None, None 
        padded_h, padded_w = None, None

    def forward(self, x):
        if not self.og_shape:
            self.og_shape = x.shape
        # print("Original x shape: ", self.og_shape)
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0) if self.padding > 0 else x
        
        if not self.pad_shape:
            self.pad_shape = x.shape
        # print("Padded x shape: ", self.pad_shape)

        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
        # print("coor shape: ", x.shape)
        x = self.flatten(x)
        # print("flattened shape: ", x.shape)

        x_dist = x[:, -2:, :]
        x = x[:, :-2, :] 

        if self.sampling_type == "all":
            similarity_matrix = self._calculate_similarity_matrix(x_dist)
            prime = self._prime(x, similarity_matrix, self.K, maximum=True)
        # print("prime shape: ", prime.shape)
        x = self.conv1d_layer(prime)
        # print("conv1d shape: ", x.shape)
        # print(x.shape)
        if not self.unflatten:
            self.unflatten = nn.Unflatten(dim=2, unflattened_size=self.pad_shape[2:])

        x = self.unflatten(x)
        # print("unflattened shape: ", x.shape)
        # print(x.shape)

        # print("final shape: ", x.shape)

        # print("sleeping for 2 seconds")
        # time.sleep(2)
        return x


    def _calculate_similarity_matrix(self, matrix, sigma=0.1):
        """Calculate similarity matrix based on coordinate distance"""
        b, c, t = matrix.shape  # c should be 2 for (x, y) coordinates

        ### TODO CHANGE IF NOT USING DISTANCE ANYMORE
        # coord_matrix = matrix[:, -2:, :]
        coord_matrix = matrix

        # Calculate pairwise Euclidean distances between coordinates
        coord_expanded_1 = coord_matrix.unsqueeze(3)  # [B, 2, T, 1]
        coord_expanded_2 = coord_matrix.unsqueeze(2)  # [B, 2, 1, T]

        # Euclidean distance between coordinates
        coord_diff = coord_expanded_1 - coord_expanded_2  # [B, 2, T, T]
        coord_dist = torch.sqrt(torch.sum(coord_diff ** 2, dim=1) + 1e-8)  # [B, T, T]
        
        # Convert distance to similarity using Gaussian kernel
        similarity_matrix = torch.exp(-coord_dist ** 2 / (2 * sigma ** 2))
    
        return similarity_matrix

    def _prime(self, matrix, magnitude_matrix, K, maximum):
        b, c, t = matrix.shape
        # _, topk_indices = torch.topk(magnitude_matrix.detach(), k=K, dim=2, largest=maximum)

        _, sorted_indices = torch.sort(magnitude_matrix.detach(), dim=2, descending=True, stable=True)
        topk_indices = sorted_indices[:, :, :K]

        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    

        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=topk_indices_exp)
        # prime, _ = self.filter_non_zero_starting_rows_multichannel(prime)
        # b, c, num_filtered_rows, k = prime.shape
        # print(prime.shape)

        if self.sample_padding > 0:
            prime = prime.view(b, c, self.pad_shape[-2], self.pad_shape[-1], K)
            # print(prime.shape)
            prime = prime[:, :, self.padding:-self.padding, self.padding:-self.padding, :]
            # print(prime.shape)

            prime = prime.reshape(b, c, K * self.og_shape[-2] * self.og_shape[-1])
        else: 
            prime = prime.view(b, c, -1)

        # print(prime.shape)
        
        return prime

    # def filter_non_zero_starting_rows_multichannel(self, tensor):
    #     """
    #     Filter rows based on the first element of the first channel being non-zero
        
    #     Args:
    #         tensor: Input tensor of shape [B, C, num_rows, row_length]
        
    #     Returns:
    #         Filtered tensor with only rows where first channel's first element != 0
    #     """
    #     # Get the shape
    #     b, c, num_rows, row_length = tensor.shape
        
    #     # Create mask based on first channel only
    #     # tensor[:, 0, :, 0] gets first element of each row in first channel
            
    #     mask = tensor[:, 0, :, 0].detach() != 0  # Shape: [b, num_rows]
        
    #     # Get indices of non-zero starting rows
    #     non_zero_indices = torch.where(mask[0])[0]  # [0] because batch dimension
    
    #     # Select rows from ALL channels
    #     filtered_tensor = tensor[:, :, non_zero_indices, :]
        
    #     return filtered_tensor, non_zero_indices

    def _add_coordinate_encoding(self, x):
        b, _, h, w = x.shape
        cache_key = f"{b}_{h}_{w}_{x.device}"

        if cache_key in self.coordinate_cache:
            expanded_grid = self.coordinate_cache[cache_key]
        else:
            with torch.no_grad():
                y_coords_vec = torch.linspace(start=-1, end=1, steps=h, device=x.device)
                x_coords_vec = torch.linspace(start=-1, end=1, steps=w, device=x.device)

                y_grid, x_grid = torch.meshgrid(y_coords_vec, x_coords_vec, indexing='ij')
                grid = torch.stack((x_grid, y_grid), dim=0).unsqueeze(0)
                expanded_grid = grid.expand(b, -1, -1, -1)
                self.coordinate_cache[cache_key] = expanded_grid

        x_with_coords = torch.cat((x, expanded_grid), dim=1)
        return x_with_coords 

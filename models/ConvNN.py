import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Conv2d_NN(nn.Module):
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

        super(Conv2d_NN, self).__init__()

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


    def forward(self, x):  
        # 1. Pixel Shuffle 
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        self.og_shape = x.shape

        # 2. Add Padding 
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
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
        
        # 4. Sampling + Similarity Calculation + Aggregation
        if self.sampling_type == "all":
            similarity_matrix = self._calculate_euclidean_matrix(x_sim) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix(x_sim)
            prime = self._prime(x, similarity_matrix, self.K, self.maximum)

        elif self.sampling_type == "random":
            rand_idx = torch.randperm(x.shape[-1], device=x.device)[:self.num_samples]
            x_sample = x_sim[:, :, rand_idx]

            similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample, sqrt=True) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(x_sim, x_sample)
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
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix)
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0.0) 
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix 
        
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
        similarity_matrix = torch.bmm(norm_matrix.transpose(2, 1), norm_matrix)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0) 
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
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    
        
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

        # Map sample indices back to original matrix positions
        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=matrix.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
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


if __name__ == "__main__":
    model = Conv2d_NN(
        in_channels=3, 
        out_channels=16, 
        K=9, 
        stride=9, 
        padding=0, 
        sampling_type="spatial", 
        num_samples=8, 
        sample_padding=0,
        shuffle_pattern="NA", 
        shuffle_scale=2, 
        magnitude_type="euclidean", 
        similarity_type="Loc", 
        aggregation_type="Col"
    )

    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print("Output shape: ", out.shape)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model)
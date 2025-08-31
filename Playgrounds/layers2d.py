import torch 
import torch.nn as nn 
import torch.nn.functional as F
import time 
import math

class Conv2d_NN_sanity(nn.Module):
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
            in_channels=self.in_channels, # + 2, ## CHANGE IF NEEDED
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
        print("Original x shape: ", self.og_shape)
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0) if self.padding > 0 else x
        
        if not self.pad_shape:
            self.pad_shape = x.shape
        print("Padded x shape: ", self.pad_shape)

        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
        print("coor shape: ", x.shape)
        x = self.flatten(x)
        print("flattened shape: ", x.shape)

        x_dist = x[:, -2:, :]
        x = x[:, :-2, :] 

        if self.sampling_type == "all":
            similarity_matrix = self._calculate_similarity_matrix(x_dist)
            prime = self._prime(x, similarity_matrix, self.K, maximum=True)
        print("prime shape: ", prime.shape)
        x = self.conv1d_layer(prime)
        print("conv1d shape: ", x.shape)
        # print(x.shape)
        if not self.unflatten:
            self.unflatten = nn.Unflatten(dim=2, unflattened_size=self.og_shape[2:])

        x = self.unflatten(x)
        print("unflattened shape: ", x.shape)
        # print(x.shape)

        print("final shape: ", x.shape)

        print("sleeping for 2 seconds")
        time.sleep(2)
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
        _, topk_indices = torch.topk(magnitude_matrix.detach(), k=K, dim=2, largest=maximum)
            
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=topk_indices_exp)
        # prime, _ = self.filter_non_zero_starting_rows_multichannel(prime)
        # b, c, num_filtered_rows, k = prime.shape
        print()
        print("With Padding Ks")
        print(prime.shape)
        print(prime)
        print()
        if self.padding > 0:
            prime = prime.view(b, c, self.pad_shape[-2], self.pad_shape[-1], K)
            print("Prime with Padded shape:")
            print(prime.shape)
            print(prime)
            print()
            prime = prime[:, :, self.padding:-self.padding, self.padding:-self.padding, :]
            print("Without Padding Ks")
            print(prime.shape)
            print(prime)

            prime = prime.reshape(b, c, K * self.og_shape[-2] * self.og_shape[-1])
        else: 
            prime = prime.view(b, c, -1)
        
        return prime

    def filter_non_zero_starting_rows_multichannel(self, tensor):
        """
        Filter rows based on the first element of the first channel being non-zero
        
        Args:
            tensor: Input tensor of shape [B, C, num_rows, row_length]
        
        Returns:
            Filtered tensor with only rows where first channel's first element != 0
        """
        # Get the shape
        b, c, num_rows, row_length = tensor.shape
        
        # Create mask based on first channel only
        # tensor[:, 0, :, 0] gets first element of each row in first channel
            
        mask = tensor[:, 0, :, 0].detach() != 0  # Shape: [b, num_rows]
        
        # Get indices of non-zero starting rows
        non_zero_indices = torch.where(mask[0])[0]  # [0] because batch dimension
    
        # Select rows from ALL channels
        filtered_tensor = tensor[:, :, non_zero_indices, :]
        
        return filtered_tensor, non_zero_indices

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

if __name__ == "__main__":
    ex = torch.Tensor(
        [
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ], 
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ], 
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ]
            ]
        ]
    )
    conv = Conv2d_NN_sanity(
        in_channels=3,
        out_channels=5,
        K=9,
        stride=9,
        padding=1,
        sampling_type="all",
        num_samples=1,
        sample_padding=0,
        shuffle_pattern="none",
        shuffle_scale=1.0,
        magnitude_type="none",
        coordinate_encoding=True
    )

    out = conv(ex)
    print(out.shape)
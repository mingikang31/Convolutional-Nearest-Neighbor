"""Modular Convolutional Nearest Neighbors Framework """
from typing import Union
import torch
import torch.nn as nn 
import torch.nn.functional as F

class ConvolutionalNearestNeighbors(nn.Module):
    def __init__(self,
                config: dict[str, Union[str, int, float, bool]],
                positional_encoding_class: nn.Module, 
                similarity_class: nn.Module, 
                aggregation_class: nn.Module, 
                sampling_class: nn.Module,                 
                 ): 
        super(ConvolutionalNearestNeighbors, self).__init__()
        self.config = config
        self.padding = config.get("padding", 0)
        self.positional_encoding = positional_encoding_class
        self.similarity = similarity_class
        self.aggregation = aggregation_class
        self.sampling = sampling_class

    def _pad_input(self, x: torch.Tensor, padding: int) -> torch.Tensor: 
        if padding == 0: return x
        
        if x.dim() == 4:  # 2D Shape (B, C, H, W)
            return F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
        elif x.dim() == 3: # 1D Shape (B, C, L)
            return F.pad(x, (padding, padding), mode='constant', value=0)
        else:
            raise ValueError("Input tensor must be either 3D (B, C, L) or 4D (B, C, H, W)")
        
    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional_encoding is not None:
            return self.positional_encoding(x)
        return x

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # 2D Shape (B, C, H, W)
            return x.view(x.size(0), x.size(1), -1)  # (B, C, H*W)
        elif x.dim() == 3:  # 1D Shape (B, C, L)
            return x
        else:
            raise ValueError("Input tensor must be either 3D (B, C, L) or 4D (B, C, H, W)")

    def _undo_reshape(self, x: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        if len(original_shape) == 4:  # 2D Shape (B, C, H, W)
            return x.view(original_shape)
        elif len(original_shape) == 3:  # 1D Shape (B, C, L)
            return x
        else:
            raise ValueError("Original shape must be either 3D (B, C, L) or 4D (B, C, H, W)")

    def _calculate_similarity(self, x: torch.Tensor) -> torch.Tensor:
        return self.similarity(x, x)

    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregation(x)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        pass # TODO implement sampling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_input(x, self.padding)

        x = self._apply_positional_encoding(x)
        
        pass # TODO implement forward pass that applies positional encoding, similarity calculation, aggregation, and sampling in the correct order based on the provided classes and dimension

    

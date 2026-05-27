"""Various positional encoding classes for use in the modular Convolutional Nearest Neighbor Framework"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from Similarity import BaseSimilarity, DotProductSimilarity, CosineSimilarity, EuclideanSimilarity, ManhattanSimilarity, MinkowskiSimilarity, MahalanobisSimilarity, PearsonSimilarity, BilinearSimilarity


class BasePositionalEncoding(ABC, nn.Module):
    """Abstract base class for positional encodings.
 
    Subclasses either
    (a) modify the *input features* before similarity computation
        (``encode_features``), or
    (b) add a *bias* directly to the similarity matrix
        (``encode_similarity``).
 
    Both hooks are called inside ``forward``; subclasses only need to
    override the relevant one.
    """
 
    def encode_features(self, X: torch.Tensor) -> torch.Tensor:
        """Return positionally-augmented features.  Default: identity."""
        return X
 
    def encode_similarity(self, S: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Add a positional bias to the similarity matrix.  Default: identity."""
        return S
 
    def forward(self, X: torch.Tensor, S: Optional[torch.Tensor] = None, H: int = 1, W: int = 1):
        X_out = self.encode_features(X)
        if S is not None:
            S_out = self.encode_similarity(S, H, W)
            return X_out, S_out
        return X_out

class RawCoordinateEncoding(BasePositionalEncoding):
    """
    Append linearly-spaced normalised coordinate channels to X. 

    1-D input [B, seq_len, d] -> [B, seq_len, d+1] (one coordinate channel)
    2-D input [B, c, H, W] -> [B, c+2, H, W] (two coordinate channels)

    Coordinates are normalised to [-1, 1] with the origin at the centre of the feature map.
    """

    def encode_features(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 3: # 1-D Case
            B, seq_len, d = X.size() 
            coords = torch.linspace(-1, 1, steps=seq_len, device=X.device)
            coords = coords.view(1, seq_len, 1).expand(B, -1, 1) # [B, seq_len, 1]
            return torch.cat([X, coords], dim=-1) # [B, seq_len, d+1]
        elif X.dim() == 4: # 2-D Case
            B, C, H, W = X.size() 
            y_coords = torch.linspace(-1, 1, steps=H, device=X.device).view(1, 1, H, 1).expand(B, 1, H, W) # [B, 1, H, W]
            x_coords = torch.linspace(-1, 1, steps=W, device=X.device).view(1, 1, 1, W).expand(B, 1, H, W) # [B, 1, H, W]
            return torch.cat([X, y_coords, x_coords], dim=1) # [B, C+2, H, W]
        else:
            raise ValueError("Input tensor must be either 3D (for 1-D data) or 4D (for 2-D data).")

class FourierFeatureEncoding(BasePositionalEncoding):
    """
    Append spatial coordinates into a high-dimensional sinusoidal space using random Fourier features.

    γ(v) = [sin(2π B v), cos(2π B v)]   where B ∈ R^{d × 2} is fixed Gaussian.
    """

    def __init__(self, d: int = 16, sigma: float = 1.0):
        super().__init__()
        self.d = d 
        B = torch.rand(d, 2) * sigma
        self.register_buffer('B', B) # [d, 2]

    def _fourier_features(self, H: int, W: int, device, dtype) -> torch.Tensor:
        y = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype) # [H]
        x = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype) # [W]
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij') # [H, W]
        v = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=-1) # [H*W, 2]
        proj = 2 * np.pi * (v @ self.B.t()) # [H*W, d]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1) # [H*W, 2d]

    def encode_features(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 3: 
            B, seq_len, d = X.size() 
            features = self._fourier_features(seq_len, 1, X.device, X.dtype)
            features = features.t().view(1, seq_len, 2 * self.d).expand(B, -1, -1) # [B, seq_len, 2d]
            return torch.cat([X, features], dim=-1) # [B, seq_len, d+2d]
        elif X.dim() == 4:
            B, C, H, W = X.size() 
            features = self._fourier_features(H, W, X.device, X.dtype).t().view(1, 2 * self.d, H, W).expand(B, -1, -1, -1) # [B, 2d, H, W]
            return torch.cat([X, features], dim=1) # [B, C+2d, H, W]
        else:
            raise ValueError("Input tensor must be either 3D (for 1-D data) or 4D (for 2-D data).")

class RelativePositionalBias(BasePositionalEncoding):
    """
    Add a learnable relative positional bias directly to the similarity matrix

    S'[i, j] = S[i, j] + B[Δi, Δj]

    where Δi, Δj is the 2-D spatial offset between positions i and j. 
    The bias table B ∈ R^{(2H-1) × (2W-1)} is shared across all heads.
    """

    def __init__(self, max_H: int = 14, max_W: int = 14):
        super().__init__()
        self.max_H = max_H
        self.max_W = max_W

        # Bias table indexed by relative offset in [-H+1, H-1] x [-W+1, W-1]
        self.bias_table = nn.Parameter(
            torch.zeros(2 * max_H - 1, 2 * max_W - 1)
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def _get_bias(self, H: int, W: int) -> torch.Tensor:
        H_range = torch.arange(H, device=self.bias_table.device)
        W_range = torch.arange(W, device=self.bias_table.device)
        grid_h, grid_w = torch.meshgrid(H_range, W_range, indexing='ij')
        coords = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=0) # [2, H, W]

        # Relative offsets [2, H*W, H*W]
        rel = coords[:, :, None] - coords[:, None, :] # [2, H*W, H*W]
        # shift to [0, 2H-2] x [0, 2W-2] for indexing into bias table
        rel[0] += H - 1
        rel[1] += W - 1

        # Lookup 
        return self.bias_table[rel[0], rel[1]] # [H*W, H*W]

    def encode_similarity(self, S: torch.Tensor, H: int, W: int) -> torch.Tensor:
        bias = self._get_bias(H, W) # [H*W, H*W]
        return S + bias.unsqueeze(0) # [B, H*W, H*W]

class ConditionalPositionalEncoding(BasePositionalEncoding):
    """
    X' = X + DWConv(X)

    A lightweight, zero-padded 2-D depthwise convolution produces implicit absolute positional anchors at the spatial boundaries. The residual connection preserves the original feature scale. 
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels, bias = True)

        nn.init.zeros_(self.conv.weight) # Initialise to zero so that the module starts as identity
        nn.init.zeros_(self.conv.bias)

    def encode_features(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 4:
            raise ValueError("ConditionalPositionalEncoding only supports 4D input tensors of shape [B, C, H, W].")
        return X + self.conv(X)

class RotaryPositionalEmbedding2D(BasePositionalEncoding):
    """
    Apply 2-D RoPE to query and key vectors so that inner product encodes relative spatial offsets.

    Feature channels are split into two equal halves; each half receives rotations derived from one spatial axis (row / column). Within each half, channel pairs are rotated by angles scaled exponentially with index. 

    * Pass Q and K *before* the similarity computation: 
        rope = RotaryPositionalEmbedding2D(dim=head_dim)
        Q_rot, K_rot = rope.rotate_qk(Q, K, H, W)
        S = torch.matmul(Q_rot, K_rot.transpose(-2, -1))/sqrt(head_dim)
    """

    def __init__(self, dim:int, base: float = 10000.0):
        super().__init__() 
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE."
        self.dim = dim 
        self.half_dim = dim // 2
        self.base = base 

    def _build_freqs(self, L: int, device, dtype) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.half_dim, 2, device=device, dtype=torch.float32) / self.half_dim)
        )      
        positions = torch.arange(L, device=device, dtype=torch.float32) # [L]
        freqs = torch.outer(positions, inv_freq) # [L, half_dim//2]
        return torch.cat([freqs, freqs], dim=-1) # [L, half_dim]

    @staticmethod 
    def _rotate_hafl(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope_1d(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # x: [B, n, half_dim], freqs: [n, half_dim]
        return x * torch.cos(freqs) + self._rotate_hafl(x) * torch.sin(freqs)

    def rotate_qk(self, Q: torch.Tensor, K: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        B, n, dim = Q.size()
        assert n == H * W, "Sequence length must match H*W for 2D RoPE."

        Q_row, Q_col = Q[..., :self.half_dim], Q[..., self.half_dim:] # [B, n, half_dim]
        K_row, K_col = K[..., :self.half_dim], K[..., self.half_dim:] # [B, n, half_dim]

        # Build frequencies for row and column dimensions
        row_ids = torch.arange(H, device=Q.device).repeat_interleave(W) # [HW]
        col_ids = torch.arange(W, device=Q.device).repeat(H) # [HW]
        row_freqs = self._build_freqs(H, Q.device, Q.dtype)[row_ids] # [HW, half_dim]
        col_freqs = self._build_freqs(W, Q.device, Q.dtype)[col_ids] # [HW, half_dim]

        Q_rot = torch.cat(
            [self._apply_rope_1d(Q_row, row_freqs), self._apply_rope_1d(Q_col, col_freqs)]
            , dim=-1) # [B, n, dim]
        K_rot = torch.cat(
            [self._apply_rope_1d(K_row, row_freqs), self._apply_rope_1d(K_col, col_freqs)]
            , dim=-1) # [B, n, dim]
        return Q_rot, K_rot

    def encode_features(self, X: torch.Tensor) -> torch.Tensor:
        return X # No modification to input features; RoPE is applied directly to Q and K before similarity computation.


if __name__ == "__main__":
    torch.manual_seed(0)
    B, n, h, H, W = 2, 16, 32, 4, 4   # batch, tokens, head_dim, spatial dims
 
    Q = torch.randn(B, n, h)
    K = torch.randn(B, n, h)
    X_2d = torch.randn(B, h, H, W)     # feature map for PE tests
 
    # ---- Similarity metrics ------------------------------------------------
    metrics = {
        "DotProduct    ": DotProductSimilarity(scale=True),
        "Cosine        ": CosineSimilarity(),
        "Euclidean     ": EuclideanSimilarity(),
        "Manhattan     ": ManhattanSimilarity(),
        "Minkowski p=3 ": MinkowskiSimilarity(p=3),
        "Chebyshev     ": MinkowskiSimilarity(p=float("inf")),
        "Mahalanobis   ": MahalanobisSimilarity(dim=h),
        "Pearson       ": PearsonSimilarity(),
        "Bilinear      ": BilinearSimilarity(dim=h),
    }
 
    print("=== Similarity matrices (shape should be [2, 16, 16]) ===")
    for name, metric in metrics.items():
        S = metric(Q, K)
        print(f"  {name}: {tuple(S.shape)}  min={S.min():.3f}  max={S.max():.3f}")
 
    # ---- Positional encodings (feature-modifying) --------------------------
    print("\n=== Positional encodings (feature map [B,C,H,W] = [2,32,4,4]) ===")
 
    pe_raw   = RawCoordinateEncoding()
    pe_four  = FourierFeatureEncoding(d=8)
    pe_cpe   = ConditionalPositionalEncoding(channels=h)
 
    for name, pe in [("RawCoords  ", pe_raw),
                     ("Fourier    ", pe_four),
                     ("CPE        ", pe_cpe)]:
        out = pe(X_2d)
        print(f"  {name}: {tuple(X_2d.shape)} → {tuple(out.shape)}")

    # ---- Relative positional bias ------------------------------------------
    print("\n=== RelativePositionalBias ===")
    S_dummy = torch.zeros(B, n, n)
    rpb = RelativePositionalBias(max_H=H, max_W=W)
    S_biased = rpb.encode_similarity(S_dummy, H=H, W=W)
    print(f"  Bias applied: {tuple(S_biased.shape)}")
 
    # ---- 2-D RoPE ----------------------------------------------------------
    print("\n=== 2-D RoPE ===")
    rope = RotaryPositionalEmbedding2D(dim=h)
    Q_rot, K_rot = rope.rotate_qk(Q, K, H=H, W=W)
    print(f"  Q_rot: {tuple(Q_rot.shape)}  K_rot: {tuple(K_rot.shape)}")
 
    print("\nAll checks passed.")

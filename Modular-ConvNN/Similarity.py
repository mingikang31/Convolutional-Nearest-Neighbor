"""Various similarity classes for use in the modular Convolutional Nearest Neighbor Framework"""

"""
1. Dot Product Similarity: sim(x, y) = x^T y
2. Cosine Similarity: sim(x, y) = (x / ||x||) · (y / ||y||)
3. Euclidean Similarity: sim(x, y) = -||x - y||
4. Manhattan Similarity: sim(x, y) = -||x - y||_1
5. Minkowski Similarity: sim(x, y) = -||x - y||_p
6. Mahalanobis Similarity: sim(x, y) = -sqrt((x - y)^T Sigma^{-1} (x - y))
7. Pearson Similarity: sim(x, y) = cov(x, y) / (std(x) * std(y))
8. Bilinear Similarity: sim(x, y) = x^T W y
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

class BaseSimilarity(ABC, nn.Module):
    """Abstract base class for all similarity / distance metrics.

    Every subclass must implement ``compute(Q, K) -> Tensor`` where Q and K
    are already-projected (and optionally normalised) query / key matrices of
    shape ``[B, n, h]`` or ``[n, h]``.  A *higher* value should always mean
    *more similar* so that ``torch.topk(..., largest=True)`` returns nearest
    neighbours uniformly across all metrics.  Distance-based metrics therefore
    return the *negated* distance.
    """

    @abstractmethod
    def compute(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Return a similarity matrix S of shape ``[..., n, n]``."""
        ...

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        return self.compute(Q, K)

class DotProductSimilarity(BaseSimilarity):
    """
    sim(x, y) = x^T y

    Standard (scaled/unscaled) dot product in vanilla self-attention. Optionally applies the 1/sqrt(d) scaling factor from "Attention is All You Need".
    """

    def __init__(self, scale: bool = True):
        super().__init__()
        self.scale = scale 

    def compute(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        S = torch.matmul(Q, K.transpose(-2, -1))
        if self.scale:
            S = S / np.sqrt(Q.size(-1))
        return S

class CosineSimilarity(BaseSimilarity):
    """
    sim(x, y) = (x / ||x||) · (y / ||y||)

    Equivalent to DotProductSimilarity with L2-normalised vectors. Default metric used in ConvNN
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def compute(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        # Normalize Q and K to unit length
        Q_norm = F.normalize(Q, p=2, dim=-1, eps=self.eps)
        K_norm = F.normalize(K, p=2, dim=-1, eps=self.eps)
        # Compute cosine similarity as the dot product of normalized vectors
        return torch.matmul(Q_norm, K_norm.transpose(-2, -1))

class EuclideanSimilarity(BaseSimilarity):
    """
    sim(x, y) = -||x - y||_2

    Computed efficiently using the identity ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x^T y, to avoid an explicit O(n^2) pairwise distance calculation. 
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def compute(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q_sq = torch.sum(Q ** 2, dim=-1, keepdim=True)  # [B, n, 1]
        K_sq = torch.sum(K ** 2, dim=-1, keepdim=True)
        cross_term = torch.matmul(Q, K.transpose(-2, -1))  # [B, n, n]
        dist_sq = Q_sq - 2 * cross_term + K_sq.transpose(-2, -1)  # [B, n, n]
        dist_sq = torch.clamp(dist_sq, min=0.0)  # Numerical stability
        return -torch.sqrt(dist_sq + self.eps)

class ManhattanSimilarity(BaseSimilarity):
    """
    sim(x, y) = -||x - y||_1

    Computed using an explicit pairwise distance calculation, which is O(n^2) in the number of vectors. 
    """

    def compute(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        # Compute pairwise L1 distances
        Q_expanded = Q.unsqueeze(-2)  # [B, n, 1, h]
        K_expanded = K.unsqueeze(-3)  # [B, 1, n, h]
        dist = torch.abs(Q_expanded - K_expanded).sum(dim=-1)  # [B, n, n]
        return -dist

class MinkowskiSimilarity(BaseSimilarity):
    """
    sim(x, y) = -||x - y||_p

    p = 1 - Manhattan 
    p = 2 - Euclidean
    p = inf - Chebyshev (max absolute difference across dimensions)
    """

    def __init__(self, p: float = float('inf')):
        super().__init__()
        self.p = p

    def compute(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q_expanded = Q.unsqueeze(-2)  # [B, n, 1, h]
        K_expanded = K.unsqueeze(-3)  # [B, 1, n, h]
        dist = torch.norm(Q_expanded - K_expanded, p=self.p, dim=-1)  # [B, n, n]
        return -dist
    
    
class MahalanobisSimilarity(BaseSimilarity):
    """
    sim(x, y) = -sqrt((x - y)^T Sigma^{-1} (x - y))

    THe precision matrix E = Sigma^{-1} is learnable parameter initialised to identity. Only the diagonal of E is used for efficiency, so this is equivalent to learning a separate scaling factor for each dimension of the input space.
    """

    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__() 
        self.L = nn.Parameter(torch.eye(dim))  # Learnable Cholesky factor of precision matrix
        self.eps = eps

    def _precision(self) -> torch.Tensor:
        # Compute precision matrix E = L^T L to ensure it's positive definite
        L = torch.tril(self.L)  # Ensure L is lower triangular
        return L @ L.transpose(-2, -1)

    def compute(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        E = self._precision() # [h, h]
        Q_expanded = Q.unsqueeze(-2)  # [B, n, 1, h]
        K_expanded = K.unsqueeze(-3)  # [B, 1, n
        diff = Q_expanded - K_expanded  # [B, n, n, h]
        dist_sq = torch.einsum('...i,ij,...j->...', diff, E, diff)  # [B, n, n]
        dist_sq = torch.clamp(dist_sq, min=0.0)  # Numerical stability
        return -torch.sqrt(dist_sq + self.eps)


class PearsonSimilarity(BaseSimilarity):
    """
    sim(x, y) = cov(x, y) / (std(x) * std(y))

    Mean centered cosine similarity: centers each vector by subtracting its own mean across the h feature dimensions, then applies cosine similarity.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps 

    def compute(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q_c = Q - Q.mean(dim=-1, keepdim=True)  # Center each vector by its own mean
        K_c = K - K.mean(dim=-1, keepdim=True)
        Q_norm = F.normalize(Q_c, p=2, dim=-1, eps=self.eps)
        K_norm = F.normalize(K_c, p=2, dim=-1, eps=self.eps)
        return torch.matmul(Q_norm, K_norm.transpose(-2, -1))

class BilinearSimilarity(BaseSimilarity):
    """
    sim(x, y) = x^T W y

    Introduces a learnable weight matrrix W ∈ R^{h x h}, reducing to standard dot-product attention when W = I and to cosine attention when Q, K are L2-normalised. 
    """

    def __init__(self, dim: int, scale: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))  # Learnable weight matrix
        self.scale = scale
        self.dim = dim 

    def compute(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        QW = Q @ self.W  # [B, n, h]
        S = torch.matmul(QW, K.transpose(-2, -1))  # [B, n, n]
        if self.scale:
            S = S / np.sqrt(self.dim)
        return S
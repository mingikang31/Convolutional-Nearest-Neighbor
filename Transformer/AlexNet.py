from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn


__all__ = ["AlexNet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # (64, 55, 55) 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (64, 27, 27)
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # (192, 27, 27)
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2), # (192, 13, 13)
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # (384, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # (256, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # (256, 13, 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (256, 6, 6)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) # (256, 6, 6)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), # (1, 9216)
            nn.Linear(256 * 6 * 6, 4096), # (1, 4096) 
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    alexnet = AlexNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alexnet = alexnet.to(device)
    
    # Create sample input
    x = torch.randn(1, 3, 224, 224).to(device)
    
    # Track feature extractor layers
    print("Feature Extractor Layers:")
    print(f"Input shape: {x.shape}")
    
    # First Conv + ReLU
    x = alexnet.features[0](x)
    print(f"After Conv1 (64 filters, k=11, s=4): {x.shape}")
    x = alexnet.features[1](x)  # ReLU doesn't change shape
    
    # First MaxPool
    x = alexnet.features[2](x)
    print(f"After MaxPool1 (k=3, s=2): {x.shape}")
    
    # Second Conv + ReLU
    x = alexnet.features[3](x)
    print(f"After Conv2 (192 filters, k=5, p=2): {x.shape}")
    x = alexnet.features[4](x)  # ReLU
    
    # Second MaxPool
    x = alexnet.features[5](x)
    print(f"After MaxPool2 (k=3, s=2): {x.shape}")
    
    # Third Conv + ReLU
    x = alexnet.features[6](x)
    print(f"After Conv3 (384 filters, k=3, p=1): {x.shape}")
    x = alexnet.features[7](x)  # ReLU
    
    # Fourth Conv + ReLU
    x = alexnet.features[8](x)
    print(f"After Conv4 (256 filters, k=3, p=1): {x.shape}")
    x = alexnet.features[9](x)  # ReLU
    
    # Fifth Conv + ReLU
    x = alexnet.features[10](x)
    print(f"After Conv5 (256 filters, k=3, p=1): {x.shape}")
    x = alexnet.features[11](x)  # ReLU
    
    # Third MaxPool
    x = alexnet.features[12](x)
    print(f"After MaxPool3 (k=3, s=2): {x.shape}")
    
    # Adaptive average pooling
    x = alexnet.avgpool(x)
    print(f"After AdaptiveAvgPool (6x6): {x.shape}")
    
    # Flatten for fully connected layers
    x = torch.flatten(x, 1)
    print(f"After Flatten: {x.shape}")
    
    # Classifier layers
    print("\nClassifier Layers:")
    x = alexnet.classifier[0](x)  # Dropout doesn't change shape
    print(f"After Dropout: {x.shape}")
    
    x = alexnet.classifier[1](x)  # First FC
    print(f"After FC1 (4096 neurons): {x.shape}")
    
    x = alexnet.classifier[2](x)  # ReLU
    x = alexnet.classifier[3](x)  # Dropout
    
    x = alexnet.classifier[4](x)  # Second FC
    print(f"After FC2 (4096 neurons): {x.shape}")
    
    x = alexnet.classifier[5](x)  # ReLU
    
    x = alexnet.classifier[6](x)  # Output FC
    print(f"After FC3 (1000 neurons): {x.shape}")
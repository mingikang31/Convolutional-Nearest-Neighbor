"""
Simplified ResNet implementation using InstanceNorm2d
"""

from typing import Optional
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch import Tensor

from models.layers2d import (
    Conv2d_New,
    Conv2d_NN, 
    Conv2d_NN_Attn
)


__all__ = [
    "ResNet",
    "BasicBlock",
    "resnet18",
    "resnet34",
]


class BasicBlock(nn.Module):

    def __init__(
        self,
        args, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.args = args 
        
        
        if args.layer == "Conv2d":
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.InstanceNorm2d(out_channels)
            self.downsample = downsample
            self.stride = stride
            
        elif args.layer == "ConvNN":
            layer_params = {
                "shuffle_pattern": args.shuffle_pattern,
                "shuffle_scale": args.shuffle_scale,
                "K": args.K,
                "stride": args.K,  # Use the actual stride parameter, not K
                "sampling_type": args.sampling_type,
                "num_samples": args.num_samples,
                "sample_padding": args.sample_padding,
                "magnitude_type": args.magnitude_type,
                "coordinate_encoding": args.coordinate_encoding
            }
            self.conv1 = Conv2d_NN(in_channels, out_channels, **layer_params)
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
            self.conv2 = Conv2d_NN(out_channels, out_channels, **layer_params)
            self.bn2 = nn.InstanceNorm2d(out_channels)
            self.downsample = downsample
            self.stride = stride
            
        elif args.layer == "ConvNN_Attn":
            layer_params = {
                "shuffle_pattern": args.shuffle_pattern,
                "shuffle_scale": args.shuffle_scale,
                "K": args.K,
                "stride": args.K,  # Use the actual stride parameter, not K
                "sampling_type": args.sampling_type,
                "num_samples": args.num_samples,
                "sample_padding": args.sample_padding,
                "magnitude_type": args.magnitude_type,
                "img_size": args.img_size[1:],  # Pass H, W
                "attention_dropout": args.attention_dropout,
                "coordinate_encoding": args.coordinate_encoding
            }
            self.conv1 = Conv2d_NN_Attn(in_channels, out_channels, **layer_params)
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
            self.conv2 = Conv2d_NN_Attn(out_channels, out_channels, **layer_params)
            self.bn2 = nn.InstanceNorm2d(out_channels)
            self.downsample = downsample
            self.stride = stride
            

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        args, 
        block: BasicBlock,
        layers: list[int],
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()

        self.args = args 
        self.num_classes = args.num_classes

        self.name == f"ResNet {args.layer}"

        
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(args, block, 64, layers[0])
        self.layer2 = self._make_layer(args, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(args, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(args, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)


    def _make_layer(
        self,
        args,  # Added args parameter
        block: BasicBlock,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels),
            )

        layers = []
        layers.append(
            block(args, self.in_channels, out_channels, stride, downsample)  # Added args parameter
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(args, self.in_channels, out_channels)  # Added args parameter
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Factory functions for creating different ResNet variants
def resnet18(args) -> ResNet:
    """ResNet-18 model"""
    return ResNet(args, BasicBlock, [2, 2, 2, 2])


def resnet34(args) -> ResNet:
    """ResNet-34 model"""
    return ResNet(args, BasicBlock, [3, 4, 6, 3])


if __name__ == "__main__": 
    # Example for Conv2d
    args_conv2d = SimpleNamespace(
        layer="Conv2d",
        num_classes=100,
    )
    
    # Example for ConvNN
    args_convnn = SimpleNamespace(
        layer="ConvNN",
        K=3,
        sampling_type="all",
        num_samples=-1,
        sample_padding=0,
        shuffle_pattern="BA",
        shuffle_scale=2,
        magnitude_type="similarity",
        coordinate_encoding=False, 
        num_classes=100,
    )
    
    # Example for ConvNN_Attn
    args_convnn_attn = SimpleNamespace(
        layer="ConvNN_Attn",
        K=3,
        sampling_type="all",
        num_samples=-1,
        sample_padding=0,
        shuffle_pattern="BA",
        shuffle_scale=2,
        magnitude_type="similarity",
        coordinate_encoding=False,
        img_size=(3, 32, 32), 
        attention_dropout=0.1,
        num_classes=100,
    )
    
    # Test different configurations
    resnet_conv2d = resnet34(args_conv2d)
    resnet_convnn = resnet18(args_convnn)
    resnet_convnn_attn = resnet18(args_convnn_attn)
    
    print("Conv2d ResNet-18 parameters:", sum(p.numel() for p in resnet_conv2d.parameters() if p.requires_grad))
    print("ConvNN ResNet-18 parameters:", sum(p.numel() for p in resnet_convnn.parameters() if p.requires_grad))
    print("ConvNN_Attn ResNet-18 parameters:", sum(p.numel() for p in resnet_convnn_attn.parameters() if p.requires_grad))


"""
Conv2d ResNet-18 parameters: 21,318,948
ConvNN ResNet-18 parameters: 60,226,052
ConvNN_Attn ResNet-18 parameters: 64,420,356
"""
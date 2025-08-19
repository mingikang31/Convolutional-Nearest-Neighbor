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
    Conv2d_New_1d,
    Conv2d_NN, 
    Conv2d_NN_Attn
)


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
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        elif args.layer == "Conv2d_New": 
            layer_params = {
                "kernel_size": args.kernel_size,
                "stride": 1, 
                "shuffle_pattern": args.shuffle_pattern, 
                "shuffle_scale": args.shuffle_scale,
                "coordinate_encoding": args.coordinate_encoding
            }

            self.conv1 = Conv2d_New(in_channels, out_channels, **layer_params)
            self.conv2 = Conv2d_New(out_channels, out_channels, **layer_params)
            
        elif args.layer == "Conv2d_New_1d":
            layer_params = {
                "K": args.K, 
                "stride": 1, 
                "shuffle_pattern": args.shuffle_pattern, 
                "shuffle_scale": args.shuffle_scale,
                "coordinate_encoding": args.coordinate_encoding
            }

            self.conv1 = Conv2d_New_1d(in_channels, out_channels, **layer_params)
            self.conv2 = Conv2d_New_1d(out_channels, out_channels, **layer_params)

        elif args.layer == "ConvNN":
            layer_params = {
                "K": args.K, 
                "stride": args.K, 
                "sampling_type": args.sampling_type, 
                "num_samples": args.num_samples,
                "sample_padding": args.sample_padding,
                "shuffle_pattern": args.shuffle_pattern,
                "shuffle_scale": args.shuffle_scale,
                "magnitude_type": args.magnitude_type,
                "coordinate_encoding": args.coordinate_encoding
            }

            self.conv1 = Conv2d_NN(in_channels, out_channels, **layer_params)            
            self.conv2 = Conv2d_NN(out_channels, out_channels, **layer_params)
            
        elif args.layer == "ConvNN_Attn":
            layer_params = {
                "K": args.K, 
                "stride": args.K, 
                "sampling_type": args.sampling_type, 
                "num_samples": args.num_samples, 
                "sample_padding": args.sample_padding,
                "shuffle_pattern": args.shuffle_pattern,
                "shuffle_scale": args.shuffle_scale, 
                "magnitude_type": args.magnitude_type, 
                "img_size": args.img_size[1:], 
                "attention_dropout": args.attention_dropout, 
                "coordinate_encoding": args.coordinate_encoding
            }

            self.conv1 = Conv2d_NN_Attn(in_channels, out_channels, **layer_params)            
            self.conv2 = Conv2d_NN_Attn(out_channels, out_channels, **layer_params)

        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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

        self.name = f"ResNet {args.layer}"

        
        
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
    
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

# Factory functions for creating different ResNet variants
def resnet18(args) -> ResNet:
    """ResNet-18 model"""
    return ResNet(args, BasicBlock, [2, 2, 2, 2])


def resnet34(args) -> ResNet:
    """ResNet-34 model"""
    return ResNet(args, BasicBlock, [3, 4, 6, 3])

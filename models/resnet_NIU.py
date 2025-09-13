"""
Simplified ResNet implementation with ConvNN support
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
            self.downsample_conv1 = nn.Identity()
            self.downsample_conv2 = nn.Identity()

        elif args.layer == "Conv2d_New": 
            layer_params_1 = {
                "kernel_size": args.kernel_size,
                "stride": stride,  # Use actual stride for conv1
                "shuffle_pattern": args.shuffle_pattern, 
                "shuffle_scale": args.shuffle_scale,
                "coordinate_encoding": args.coordinate_encoding
            }
            layer_params_2 = {
                "kernel_size": args.kernel_size,
                "stride": 1,  # Always 1 for conv2
                "shuffle_pattern": args.shuffle_pattern, 
                "shuffle_scale": args.shuffle_scale,
                "coordinate_encoding": args.coordinate_encoding
            }

            self.conv1 = Conv2d_New(in_channels, out_channels, **layer_params_1)
            self.conv2 = Conv2d_New(out_channels, out_channels, **layer_params_2)
            self.downsample_conv1 = nn.Identity()
            self.downsample_conv2 = nn.Identity()

        elif args.layer == "ConvNN":
            # ConvNN layers always use stride=K and maintain spatial dimensions
            layer_params = {
                "K": args.K, 
                "stride": args.K,  # Keep as K since that's what ConvNN needs
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
            
            # Add explicit downsampling if stride > 1
            if stride > 1:
                self.downsample_conv1 = nn.AvgPool2d(kernel_size=stride, stride=stride)
            else:
                self.downsample_conv1 = nn.Identity()
            
            self.downsample_conv2 = nn.Identity()  # conv2 never downsamples
            
        elif args.layer == "ConvNN_Attn":
            # ConvNN_Attn layers always use stride=K and maintain spatial dimensions
            layer_params = {
                "K": args.K, 
                "stride": args.K,  # Keep as K since that's what ConvNN_Attn needs
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
            
            # Add explicit downsampling if stride > 1
            if stride > 1:
                self.downsample_conv1 = nn.AvgPool2d(kernel_size=stride, stride=stride)
            else:
                self.downsample_conv1 = nn.Identity()
            
            self.downsample_conv2 = nn.Identity()  # conv2 never downsamples

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample 
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.downsample_conv1(out)  # Apply downsampling after conv1 if needed
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.downsample_conv2(out)  # Apply downsampling after conv2 if needed (usually Identity)
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
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(args, block, 64, layers[0])
        self.layer2 = self._make_layer(args, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(args, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(args, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last norm layer in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        args,
        block: BasicBlock,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(
            block(args, self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(args, self.in_channels, out_channels)
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


if __name__ == "__main__":
    # Test with ConvNN
    args = SimpleNamespace(
        layer="ConvNN",
        K=3,
        sampling_type="all",
        num_samples=-1,
        sample_padding=0,
        shuffle_pattern="NA",
        shuffle_scale=2,
        magnitude_type="similarity",
        coordinate_encoding=False,
        img_size=(3, 32, 32),
        num_classes=10,
    )
    
    resnet = resnet18(args)
    print("Parameter count ResNet-18 ConvNN:", sum(p.numel() for p in resnet.parameters() if p.requires_grad))

    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    output = resnet(x)
    print(f"Output shape: {output.shape}")
    
    # Test with Conv2d
    args.layer = "Conv2d"
    resnet_conv2d = resnet18(args)
    print("Parameter count ResNet-18 Conv2d:", sum(p.numel() for p in resnet_conv2d.parameters() if p.requires_grad))
    
    output_conv2d = resnet_conv2d(x)
    print(f"Conv2d Output shape: {output_conv2d.shape}")
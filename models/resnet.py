"""
model.resnet.py 
ResNet Model Definition with Custom Convolutional Layers
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchsummary import summary 
import numpy as np

from models.layers2d import (
    Conv2d_NN, 
    Conv2d_NN_Attn,
    Conv2d_Branching, 
    Conv2d_Attn_Branching
)

"""
ResNet Model with Custom Convolutional Layers 

ResNet-18 Params: 11,689,512
ResNet-34 Params: 21,797,672
ResNet-50 Params: 25,557,032
"""
class ResNet(nn.Module):
    def __init__(self, args): 
        super(ResNet, self).__init__()
        self.args = args 
        self.num_classes = args.num_classes 

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Model Layer Configurations 
        if args.model == "resnet18":
            layers = [2, 2, 2, 2]
            block = ResBlock 
            self.expansion = 1
        elif args.model == "resnet34":
            layers = [3, 4, 6, 3]
            block = ResBlock
            self.expansion = 1
        elif args.model == "resnet50":
            layers = [3, 4, 6, 3]
            block = BottleNeck
            self.expansion = 4
        else:
            raise ValueError("Invalid model type. Choose from 'resnet18', 'resnet34', 'resnet50'")

        self.in_channels = 64 

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier Layer 
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(start_dim=1), 
            nn.Linear(512 * self.expansion, self.num_classes)
        )

        self.name = "ResNet"

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = [] 

        # First block with downsampling if needed 
        layers.append(block(self.args, self.in_channels, out_channels, stride))

        if block == BottleNeck:
            self.in_channels = out_channels * block.expansion 
        else: 
            self.in_channels = out_channels 

        # Remaining blocks use stride = 1
        for _ in range(1, blocks):
            layers.append(block(self.args, self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def forward(self, x):
        x = self.first_conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, 
                 args, 
                 in_channels, 
                 out_channels, 
                 stride=1):
        super(ResBlock, self).__init__()
        self.args = args 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        convnn_params = {
            "K": self.args.K, 
            "stride": self.args.K, # Stride is always K
            "padding": self.args.padding,
            "sampling_type": self.args.sampling_type,
            "num_samples": self.args.num_samples,
            "sample_padding": self.args.sample_padding,
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "magnitude_type": self.args.magnitude_type,
            "similarity_type": self.args.similarity_type,
            "aggregation_type": self.args.aggregation_type, 
            "lambda_param": self.args.lambda_param
        }
        
        convnn_attn_params = {
            "K": self.args.K, 
            "stride": self.args.K, # Stride is always K
            "padding": self.args.padding,
            "sampling_type": self.args.sampling_type,
            "num_samples": self.args.num_samples,
            "sample_padding": self.args.sample_padding,
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "magnitude_type": self.args.magnitude_type,
            "aggregation_type": self.args.aggregation_type,
            "attention_dropout": self.args.attention_dropout
        }

        convnn_branching_params = {
            "kernel_size": self.args.kernel_size,
            "K": self.args.K,
            "stride": self.args.K, # Stride is always K
            "padding": self.args.padding,
            "sampling_type": self.args.sampling_type,
            "num_samples": self.args.num_samples,
            "sample_padding": self.args.sample_padding,
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "magnitude_type": self.args.magnitude_type,
            "similarity_type": self.args.similarity_type,
            "aggregation_type": self.args.aggregation_type, 
            "lambda_param": self.args.lambda_param,
            "branch_ratio": self.args.branch_ratio
        }

        convnn_attn_branching_params = {
            "kernel_size": self.args.kernel_size,
            "K": self.args.K,
            "stride": self.args.K, # Stride is always K
            "padding": self.args.padding,
            "sampling_type": self.args.sampling_type,
            "num_samples": self.args.num_samples,
            "sample_padding": self.args.sample_padding,
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "magnitude_type": self.args.magnitude_type,
            "aggregation_type": self.args.aggregation_type,
            "attention_dropout": self.args.attention_dropout,
            "branch_ratio": self.args.branch_ratio
        }

        # Check Convolutional Arguments for layer 1
        if stride == 1:
            if self.args.layer == "Conv2d":
                conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            elif args.layer == "ConvNN":
                conv1 = Conv2d_NN(out_channels, out_channels, **convnn_params)
            elif args.layer == "ConvNN_Attn":
                conv1 = Conv2d_NN_Attn(out_channels, out_channels, **convnn_attn_params)
            elif args.layer == "Branching":
                conv1 = Conv2d_Branching(out_channels, out_channels, **convnn_branching_params)
            elif args.layer == "Branching_Attn":
                conv1 = Conv2d_Attn_Branching(out_channels, out_channels, **convnn_attn_branching_params)
        else: 
            conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # Layer 2 Convolution
        if self.args.layer == "Conv2d":
            conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        elif args.layer == "ConvNN":
            conv2 = Conv2d_NN(out_channels, out_channels, **convnn_params)
        elif args.layer == "ConvNN_Attn":
            conv2 = Conv2d_NN_Attn(out_channels, out_channels, **convnn_attn_params)
        elif args.layer == "Branching":
            conv2 = Conv2d_Branching(out_channels, out_channels, **convnn_branching_params)
        elif args.layer == "Branching_Attn":
            conv2 = Conv2d_Attn_Branching(out_channels, out_channels, **convnn_attn_branching_params)

            
        self.layer1 = nn.Sequential(
            conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            conv2, 
            nn.BatchNorm2d(out_channels)
        )

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, 
        #               padding=1, bias=False), 
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

        
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
        #               padding=1, bias=False), 
        #     nn.BatchNorm2d(out_channels)
        # )

        # Identity mapping 
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else: 
            self.identity = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out = self.layer1(x)
        out = self.layer2(out)

        out += identity
        out = self.relu(out)
        return out

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self,
                    args, 
                    in_channels, 
                    out_channels, 
                    stride=1):
        super(BottleNeck, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        convnn_params = {
            "K": self.args.K, 
            "stride": self.args.K, # Stride is always K
            "padding": self.args.padding,
            "sampling_type": self.args.sampling_type,
            "num_samples": self.args.num_samples,
            "sample_padding": self.args.sample_padding,
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "magnitude_type": self.args.magnitude_type,
            "similarity_type": self.args.similarity_type,
            "aggregation_type": self.args.aggregation_type, 
            "lambda_param": self.args.lambda_param
        }
        
        convnn_attn_params = {
            "K": self.args.K, 
            "stride": self.args.K, # Stride is always K
            "padding": self.args.padding,
            "sampling_type": self.args.sampling_type,
            "num_samples": self.args.num_samples,
            "sample_padding": self.args.sample_padding,
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "magnitude_type": self.args.magnitude_type,
            "aggregation_type": self.args.aggregation_type,
            "attention_dropout": self.args.attention_dropout
        }

        convnn_branching_params = {
            "kernel_size": self.args.kernel_size,
            "K": self.args.K,
            "stride": self.args.K, # Stride is always K
            "padding": self.args.padding,
            "sampling_type": self.args.sampling_type,
            "num_samples": self.args.num_samples,
            "sample_padding": self.args.sample_padding,
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "magnitude_type": self.args.magnitude_type,
            "similarity_type": self.args.similarity_type,
            "aggregation_type": self.args.aggregation_type, 
            "lambda_param": self.args.lambda_param,
            "branch_ratio": self.args.branch_ratio
        }

        convnn_attn_branching_params = {
            "kernel_size": self.args.kernel_size,
            "K": self.args.K,
            "stride": self.args.K, # Stride is always K
            "padding": self.args.padding,
            "sampling_type": self.args.sampling_type,
            "num_samples": self.args.num_samples,
            "sample_padding": self.args.sample_padding,
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "magnitude_type": self.args.magnitude_type,
            "aggregation_type": self.args.aggregation_type,
            "attention_dropout": self.args.attention_dropout,
            "branch_ratio": self.args.branch_ratio
        }
        
        # 1x1 Conv to reduce dimensions
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Check Convolutional Arguments for layer 1
        if stride == 1:
            if self.args.layer == "Conv2d":
                main_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            elif args.layer == "ConvNN":
                main_conv = Conv2d_NN(out_channels, out_channels, **convnn_params)
            elif args.layer == "ConvNN_Attn":
                main_conv = Conv2d_NN_Attn(out_channels, out_channels, **convnn_attn_params)
            elif args.layer == "Branching":
                main_conv = Conv2d_Branching(out_channels, out_channels, **convnn_branching_params)
            elif args.layer == "Branching_Attn":
                main_conv = Conv2d_Attn_Branching(out_channels, out_channels, **convnn_attn_branching_params)
        else: 
            main_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # 3x3 Conv with stride (main processing) 
        self.conv2 = nn.Sequential(
            main_conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # # 3x3 Conv with stride (main processing) 
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
        #               padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

        
        # 1x1 Conv to expand channels 
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False), 
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        # Identity mapping 
        if stride != 1 or in_channels != out_channels * self.expansion: 
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else: 
            self.identity = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += identity
        out = self.relu(out)
        return out

if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace(
        model="resnet50",
        layer="Branching_Attn",
        K=9,
        kernel_size=3,
        padding=1,
        sampling_type="all",
        num_samples=-1,
        sample_padding=0,
        shuffle_pattern="NA",
        shuffle_scale=0.0,
        magnitude_type="cosine",
        similarity_type="Col",
        aggregation_type="Col",
        lambda_param=0.5,
        attention_dropout=0.1,
        branch_ratio=0.5,
        num_classes=1000
    )

    model = ResNet(args)
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    ex = torch.randn(3, 3, 224, 224)
    out = model(ex)
    print(f"Output shape: {out.shape}")
    
    summary(model, (3, 224, 224))

    # ConvNN ResNet-50
    # Total Parameters = 25,559,912

    # 
    
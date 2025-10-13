import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from types import SimpleNamespace

from models.layers2d import (
    Conv2d_New, 
    Conv2d_NN, 
    Conv2d_Branching, 
    Conv2d_NN_Attn, 
    Conv2d_Attn_Branching
)


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.args = args 
        self.num_classes = args.num_classes
        """
        ResNet-18 Params: 11,689,512
        ResNet-34 Params: 21,797,672
        ResNet-50 Params: 25,557,032
        """
        
        self.name = f"{args.model} - {args.layer}"

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Resnet 18 - [64, 128, 256, 512] * [2, 2, 2, 2]
        # Resnet 34 - [64, 128, 256, 512] * [3, 4, 6, 3]
        # Resnet 50 - [64, 128, 256, 512] * [3, 4, 6, 3]

        if args.model == "resnet18":
            layers = [2, 2, 2, 2]
        elif args.model == "resnet34":
            layers = [3, 4, 6, 3]
        elif args.model == "resnet50":
            layers = [0, 0, 0, 0] # Placeholder for ResNet-50, needs Bottleneck implementation
        else:
            raise ValueError("Invalid model type. Choose from 'resnet18', 'resnet34', 'resnet50'")

        self.in_channels = 64

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1])
        self.layer3 = self._make_layer(256, layers[2])
        self.layer4 = self._make_layer(512, layers[3])


        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(start_dim=1),
            nn.Linear(512 * 7 * 7, self.num_classes)
        )

        

    def _make_layer(self, out_channels, blocks):
        layers = []

        # First block 
        layers.append(ResBlock(self.args, self.in_channels, out_channels))
        self.in_channels = out_channels 

        # Remaining blocks 
        for _ in range(1, blocks):
            layers.append(ResBlock(self.args, self.in_channels, out_channels))

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
                 ):

        super(ResBlock, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        
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

        if args.layer == "Conv2d":
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        elif args.layer == "ConvNN":
            self.conv1 = Conv2d_NN(in_channels, out_channels, **convnn_params)
            self.conv2 = Conv2d_NN(out_channels, out_channels, **convnn_params)
        elif args.layer == "Branching":
            self.conv1 = Conv2d_Branching(in_channels, out_channels, **convnn_branching_params)
            self.conv2 = Conv2d_Branching(out_channels, out_channels, **convnn_branching_params)


        self.layer1 = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            self.conv2,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Identity mapping
        if in_channels != out_channels:
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
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

if __name__ == "__main__":
    args = SimpleNamespace(
        layer="Conv2d",
        K=9,
        kernel_size=3, 
        padding=1,
        sampling_type="all",
        num_samples=-1,
        sample_padding=0,
        shuffle_pattern="NA",
        shuffle_scale=0,
        magnitude_type="euclidean",
        similarity_type="Loc",
        aggregation_type="Col",
        lambda_param=0.5,
        branch_ratio=0.5,
        attention_dropout=0.1,
        coordinate_encoding=False,
        model="resnet34",
        num_classes=10
    )

    model = ResNet(args)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
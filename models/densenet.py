import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers2d import (
    Conv2d_NN, 
    Conv2d_Branching, 
    Conv2d_NN_Attn, 
    Conv2d_Attn_Branching
)


class DenseNet(nn.Module):
    """
    DenseNet-121: growth_rate=32, block_config=(6, 12, 24, 16)
    DenseNet-169: growth_rate=32, block_config=(6, 12, 32, 32)
    DenseNet-201: growth_rate=32, block_config=(6, 12, 48, 32)
    DenseNet-264: growth_rate=32, block_config=(6, 12, 64, 48)
    """
    def __init__(self, args, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_features=64, bn_size=4, drop_rate=0.0):
        super(DenseNet, self).__init__()
        
        self.args = args
        self.num_classes = args.num_classes
        self.name = f"{args.model} - {args.layer}"
        
        # First convolution (for CIFAR: no stride, no pool)
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                args=args,
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = Transition(num_input_features=num_features, 
                                  num_output_features=num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        
        # Classifier
        self.classifier = nn.Linear(num_features, self.num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


class DenseBlock(nn.Module):
    def __init__(self, args, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                args=args,
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'denselayer{i+1}', layer)
    
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseLayer(nn.Module):
    def __init__(self, args, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.args = args
        
        # Setup ConvNN parameters
        convnn_params = {
            "K": self.args.K,
            "stride": self.args.K,
            "padding": self.args.padding,
            "sampling_type": getattr(self.args, 'sampling_type', 'all'),
            "num_samples": getattr(self.args, 'num_samples', -1),
            "sample_padding": getattr(self.args, 'sample_padding', 0),
            "shuffle_pattern": getattr(self.args, 'shuffle_pattern', 'NA'),
            "shuffle_scale": getattr(self.args, 'shuffle_scale', 0),
            "magnitude_type": getattr(self.args, 'magnitude_type', 'euclidean'),
            "similarity_type": getattr(self.args, 'similarity_type', 'Loc'),
            "aggregation_type": getattr(self.args, 'aggregation_type', 'Col'),
            "lambda_param": getattr(self.args, 'lambda_param', 0.5)
        }
        
        convnn_branching_params = {
            **convnn_params,
            "kernel_size": self.args.kernel_size,
            "branch_ratio": getattr(self.args, 'branch_ratio', 0.5)
        }
        
        # 1x1 conv
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                              kernel_size=1, stride=1, bias=False)
        
        # 3x3 conv (or ConvNN)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        
        if args.layer == "Conv2d":
            self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                                  kernel_size=3, stride=1, padding=1, bias=False)
        elif args.layer == "ConvNN":
            self.conv2 = Conv2d_NN(bn_size * growth_rate, growth_rate, **convnn_params)
        elif args.layer == "Branching":
            self.conv2 = Conv2d_Branching(bn_size * growth_rate, growth_rate, 
                                         **convnn_branching_params)
        elif args.layer == "ConvNN_Attn":
            convnn_attn_params = {**convnn_params}
            convnn_attn_params["attention_dropout"] = getattr(self.args, 'attention_dropout', 0.1)
            convnn_attn_params.pop("similarity_type", None)
            convnn_attn_params.pop("lambda_param", None)
            self.conv2 = Conv2d_NN_Attn(bn_size * growth_rate, growth_rate, 
                                       **convnn_attn_params)
        elif args.layer == "Branching_Attn":
            convnn_attn_branching_params = {**convnn_branching_params}
            convnn_attn_branching_params["attention_dropout"] = getattr(self.args, 'attention_dropout', 0.1)
            convnn_attn_branching_params.pop("similarity_type", None)
            convnn_attn_branching_params.pop("lambda_param", None)
            self.conv2 = Conv2d_Attn_Branching(bn_size * growth_rate, growth_rate,
                                              **convnn_attn_branching_params)
        
        self.drop_rate = drop_rate
    
    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class Transition(nn.Module):
    """Transition layer between dense blocks - uses pooling for downsampling"""
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out


if __name__ == "__main__":
    from types import SimpleNamespace
    
    args = SimpleNamespace(
        model="densenet121",
        layer="Conv2d",
        num_classes=100,
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
        attention_dropout=0.1
    )
    
    model = DenseNet(args, growth_rate=32, block_config=(6, 12, 24, 16))
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Output shape: {y.shape}")
    total, trainable = model.parameter_count()
    print(f"Total params: {total:,}, Trainable: {trainable:,}")
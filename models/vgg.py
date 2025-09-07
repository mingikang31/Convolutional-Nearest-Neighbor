r"""
@software{torchvision2016,
    title        = {TorchVision: PyTorch's Computer Vision library},
    author       = {TorchVision maintainers and contributors},
    year         = 2016,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/pytorch/vision}}
}
"""

import torch 
import torch.nn as nn 

from models.layers2d import (
    Conv2d_New, 
    Conv2d_NN, 
    Conv2d_NN_Attn
)


class VGG(nn.Module):
    def __init__(
        self, 
        args,
        # in_channels=3, 
        features_config="A", 
        # num_classes=1000,
        dropout=0.5    
    ):
        super(VGG, self).__init__()
        """
        A: VGG-11 Params: 132,868,840
        B: VGG-13 Params: 133,053,736
        D: VGG-16 Params: 138,365,992
        E: VGG-19 Params: 143,678,248
        features_config: str, one of "A", "B", "D", "E"
        """

        self.args = args 
        in_channels = self.args.img_size[0] 
        num_classes = self.args.num_classes

        self.name = f"VGG {features_config} {args.layer}"

        cfg = {
            "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }

        layers = [] 

        for v in cfg[features_config]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layer_params = {
                    "in_channels": in_channels,
                    "out_channels": v,
                    "shuffle_pattern": self.args.shuffle_pattern,
                    "shuffle_scale": self.args.shuffle_scale,
                }
                
                if self.args.layer == "Conv2d":
                    layer = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    
                elif args.layer == "Conv2d_New": 
                    layer_params.update({
                        "kernel_size": args.kernel_size,
                        "stride": 1, 
                        "shuffle_pattern": args.shuffle_pattern, 
                        "shuffle_scale": args.shuffle_scale,
                        "coordinate_encoding": args.coordinate_encoding
                    })
                    layer = Conv2d_New(**layer_params)

                elif self.args.layer == "ConvNN":
                    layer_params.update({
                        "K": self.args.K,
                        "stride": self.args.K, # Stride is always K
                        "sampling_type": self.args.sampling_type,
                        "num_samples": self.args.num_samples,
                        "sample_padding": self.args.sample_padding,
                        "magnitude_type": self.args.magnitude_type,
                        "coordinate_encoding": self.args.coordinate_encoding
                    })
                    layer = Conv2d_NN(**layer_params)
                elif self.args.layer == "ConvNN_Attn":
                    layer_params.update({
                        "K": self.args.K,
                        "stride": self.args.K,
                        "sampling_type": self.args.sampling_type,
                        "num_samples": self.args.num_samples,
                        "sample_padding": self.args.sample_padding,
                        "magnitude_type": self.args.magnitude_type,
                        "img_size": self.args.img_size[1:], # Pass H, W
                        "attention_dropout": self.args.attention_dropout,
                        "coordinate_encoding": self.args.coordinate_encoding
                    })
                    layer = Conv2d_NN_Attn(**layer_params)
                    
                layers += [layer]
                layers += [nn.BatchNorm2d(v)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = v

        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
        
if __name__ == "__main__":
    from types import SimpleNamespace

    # Create default args
    args = SimpleNamespace(
        layer="ConvNN",
        num_layers=3,
        channels=[8, 16, 32],
        K=3,
        sampling_type="all",
        num_samples=-1,
        sample_padding=0,
        num_heads=4,
        attention_dropout=0.1,
        shuffle_pattern="NA",
        shuffle_scale=2,
        magnitude_type="similarity",
        coordinate_encoding=True, 
        img_size=(3, 32, 32), 
        num_classes=10,
    )
    model = VGG(
        args=args,
        features_config="A",
    )
    print("Parameter count ConvNN: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    args.layer = "Conv2d"
    model = VGG(
        args=args,
        features_config="A",
    )
    print("Parameter count Conv2d: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    args.layer = "ConvNN_Attn"
    model = VGG(
        args=args,
        features_config="A",
    )
    print("Parameter count ConvNN_Attn: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


"""
Parameter count Conv2d:  128,807,306
Parameter count ConvNN:  170,448,554
Parameter count ConvNN_Attn:  172,545,706
"""

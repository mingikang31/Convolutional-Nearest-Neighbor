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
    Conv2d_NN_Attn, 
    Conv2d_Branching
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

        conv2d_new_params = {
            "kernel_size": self.args.kernel_size,
            "stride": 1, # Stride is always 1 
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "aggregation_type": self.args.aggregation_type
        }

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

            
        for v in cfg[features_config]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.args.layer == "Conv2d":
                    layer = nn.Conv2d(in_channels, 
                                      v, 
                                      kernel_size=self.args.kernel_size, 
                                      stride=1, 
                                      padding="same", 
                                      )
                elif args.layer == "Conv2d_New": 
                    conv2d_new_params.update({
                        "in_channels": in_channels,
                        "out_channels": v
                    })
                    layer = Conv2d_New(**conv2d_new_params)

                elif self.args.layer == "ConvNN":
                    convnn_params.update({
                        "in_channels": in_channels,
                        "out_channels": v,
                    })
                    layer = Conv2d_NN(**convnn_params)
                elif self.args.layer == "ConvNN_Attn":
                    convnn_attn_params.update({
                        "in_channels": in_channels,
                        "out_channels": v,

                    })
                    layer = Conv2d_NN_Attn(**convnn_attn_params)
                elif self.args.layer == "Branching":
                    convnn_branching_params.update({
                        "in_channels": in_channels,
                        "out_channels": v,
                    })
                    layer = Conv2d_Branching(**convnn_branching_params)

                ## Changed to Instance Norm to go first then conv (different from VGG paper) 
                """
                original:
                layers += [layer]
                layers += [nn.BatchNorm2d(v)]
                layers += [nn.ReLU(inplace=True)]
                """
                
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
    pass

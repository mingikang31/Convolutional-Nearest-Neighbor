import torch 
import torch.nn as nn
from torchsummary import summary


from models.layers2d import (
    Conv2d_New,
    Conv2d_NN, 
    Conv2d_NN_Attn
)


class AllConvNet(nn.Module): 
    def __init__(self, args): 
        super(AllConvNet, self).__init__()
        self.args = args
        self.model = "All Convolutional Network"
        self.name = f"{self.model} {self.args.layer}"
        
        layers = []
        in_ch = self.args.img_size[0] 

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
            "aggregation_type": self.args.aggregation_type
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
            "similarity_type": self.args.similarity_type,
            "aggregation_type": self.args.aggregation_type, 
            
            "attention_dropout": self.args.attention_dropout
        }
        
        for i in range(self.args.num_layers):
            out_ch = self.args.channels[i]

            if self.args.layer == "Conv2d":
                layer = nn.Conv2d(
                    in_channels=in_ch, 
                    out_channels=out_ch, 
                    kernel_size=self.args.kernel_size, 
                    stride=1, 
                    padding='same', 
                    bias=False
                )
            elif self.args.layer == "Conv2d_New":
                conv2d_new_params.update({
                    "in_channels": in_ch,
                    "out_channels": out_ch
                })
                layer = Conv2d_New(**conv2d_new_params)

            elif self.args.layer == "ConvNN":
                convnn_params.update({
                    "in_channels": in_ch,
                    "out_channels": out_ch
                })
                layer = Conv2d_NN(**convnn_params)

            elif self.args.layer == "ConvNN_Attn":
                convnn_attn_params.update({
                    "in_channels": in_ch,
                    "out_channels": out_ch
                })
                layer = Conv2d_NN_Attn(**convnn_attn_params)
            

            layers.append(nn.InstanceNorm2d(num_features=out_ch)) # Pre-layer normalization
            layers.append(layer)
            layers.append(nn.ReLU(inplace=True))
            
            # Update in_ch for the next layer
            in_ch = out_ch
            
        self.features = nn.Sequential(*layers)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
            
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_ch, self.args.num_classes) # Use the final in_ch value
        )
        
        self.to(self.args.device)

    def forward(self, x): 
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def summary(self): 
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but img_size doesn't include it
            summary(self, input_size=self.img_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)
        
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

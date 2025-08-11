import math
from typing import List, Tuple
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F


from layers2d import (
    Conv2d_NN, 
    Conv2d_NN_Attn
)

class Unet(nn.Module):

    def __init__(
        self,
        args, 
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
       
        super().__init__()

        self.args = args
        

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(args, in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(args, ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(args, ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(self.args, ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(args,ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, args, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()
        self.args = args


        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        layer_params = {
                    "shuffle_pattern": self.args.shuffle_pattern,
                    "shuffle_scale": self.args.shuffle_scale,
                    "K": self.args.K,
                    "sampling_type": self.args.sampling_type,
                    "num_samples": self.args.num_samples,
                    "sample_padding": self.args.sample_padding,
                    "num_heads": self.args.num_heads,
                    "attention_dropout": self.args.attention_dropout,
                    "magnitude_type": self.args.magnitude_type,
                    "coordinate_encoding": self.args.coordinate_encoding
                    
        }

        if self.args.layer == "Conv2d":
            self.layers = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
            )
        elif self.args.layer == "ConvNN":
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
            
            self.layers = nn.Sequential(
                Conv2d_NN(in_chans, out_chans, **layer_params),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
                Conv2d_NN(out_chans, out_chans, **layer_params),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
            )
        elif self.args.layer == "ConvNN_Attn":
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
            self.layers = nn.Sequential(
                Conv2d_NN(in_chans, out_chans, **layer_params),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
                Conv2d_NN(out_chans, out_chans, **layer_params),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
            )
            

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        
        self.layers = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), 
        #     nn.Conv2d(
        #         in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False
        #     ),
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)

if __name__ == "__main__":  
     # Create default args
    args = SimpleNamespace(
        layer="ConvNN",
        num_layers=3,
        channels=[8, 16, 32],
        K=1,
        sampling_type="all",
        num_samples=-1,
        sample_padding=0,
        num_heads=4,
        attention_dropout=0.1,
        shuffle_pattern="BA",
        shuffle_scale=2,
        magnitude_type="similarity",
        coordinate_encoding=True, 
        img_size=(3, 32, 32), 
        num_classes=10,
    )
    model = Unet(
        args=args,
        in_chans=3,
        out_chans=3,
        chans=32,
        num_pool_layers=3,
        drop_prob=0.1
    )
    print("Parameter count ConvNN: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    args.layer = "Conv2d"
    model = Unet(
        args=args,
        in_chans=3,
        out_chans=3,
        chans=32,
        num_pool_layers=3,
        drop_prob=0.1
    )
    print("Parameter count Conv2d: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
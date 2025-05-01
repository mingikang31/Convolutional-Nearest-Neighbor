import torch 
import torch.nn as nn
from torchsummary import summary


from layers2d import (
    Conv2d_NN,
    Conv2d_NN_Spatial,
    Conv2d_NN_Attn,
    Conv2d_NN_Attn_Spatial,
    Attention2d,
    Conv2d_ConvNN_Branching,
    Conv2d_ConvNN_Spatial_Branching,
    Conv2d_ConvNN_Attn_Branching,
    Conv2d_ConvNN_Attn_Spatial_Branching,
    Attention_ConvNN_Branching,
    Attention_ConvNN_Spatial_Branching,
    Attention_ConvNN_Attn_Branching,
    Attention_ConvNN_Attn_Spatial_Branching,
    Attention_Conv2d_Branching,    
    
)

class ClassificationModel(nn.Module): 
    def __init__(self, args): 
        super(ClassificationModel, self).__init__()
        self.args = args
        self.model = args.model 
        
        # Model Parameters
        self.k_kernel = int(args.k_kernel)
        self.sampling = args.sampling
        self.shuffle_pattern = args.shuffle_pattern
        self.shuffle_scale = int(args.shuffle_scale)
        self.magnitude_type = args.magnitude_type
        self.location_channels = args.location_channels
        self.num_heads = int(args.num_heads)
        
        self.num_samples = int(args.num_samples) if args.num_samples != "all" else "all"
        self.num_classes = int(args.num_classes)
        self.device = args.device
        
        # In Channels, Middle Channels, and Number of Layers
        self.img_size = args.img_size
        self.in_ch = int(self.img_size[0]) # Number of Channels
        self.mid_ch = int(args.hidden_dim) # Hidden Dimension
        self.num_layers = int(args.num_layers)
        
        
        
        
        assert self.num_layers >= 2, "Number of layers must be at least 2"
        assert self.mid_ch >= 8, "Middle channels must be at least 8"

        layers = [] 
        
        for i in range(self.num_layers):
            if i == 0: 
                if args.model == "Conv2d":
                    layers.append(
                        nn.Conv2d(
                            in_channels=self.in_ch, 
                            out_channels=self.mid_ch, 
                            kernel_size=self.k_kernel, 
                            stride=1, 
                            padding=(self.k_kernel - 1) // 2 if self.k_kernel % 2 == 1 else self.k_kernel // 2
                        )
                    )
                elif args.model == "ConvNN": 
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Conv2d_NN(
                                in_channels=self.in_ch, 
                                out_channels=self.mid_ch, 
                                K=self.k_kernel,
                                stride=self.k_kernel, 
                                padding=0, 
                                shuffle_pattern=self.shuffle_pattern, 
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type,
                                location_channels=self.location_channels
                            )
                        )
                    elif self.sampling == "Spatial":
                        layers.append(
                            Conv2d_NN_Spatial(
                                in_channels=self.in_ch, 
                                out_channels=self.mid_ch, 
                                K=self.k_kernel,
                                stride=self.k_kernel, 
                                padding=0, 
                                shuffle_pattern=self.shuffle_pattern, 
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                sample_padding=0,
                                magnitude_type=self.magnitude_type,
                                location_channels=self.location_channels
                            )
                        )
                elif args.model == "ConvNN_Attn":
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Conv2d_NN_Attn(
                                in_channels=self.in_ch, 
                                out_channels=self.mid_ch, 
                                K=self.k_kernel,
                                stride=self.k_kernel, 
                                padding=0, 
                                shuffle_pattern=self.shuffle_pattern, 
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type,
                                location_channels=self.location_channels, 
                                image_size=self.img_size[1:]
                            )
                        )   
                    elif self.sampling == "Spatial":
                        layers.append(
                            Conv2d_NN_Attn_Spatial(
                                in_channels=self.in_ch, 
                                out_channels=self.mid_ch, 
                                K=self.k_kernel,
                                stride=self.k_kernel, 
                                padding=0, 
                                shuffle_pattern=self.shuffle_pattern, 
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type,
                                location_channels=self.location_channels, 
                                image_size=self.img_size[1:],
                            )   
                        )
                elif args.model == "Attention":
                    layers.append(
                        Attention2d(
                            in_channels=self.in_ch, 
                            out_channels=self.mid_ch,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels,
                        )
                    )
                elif args.model == "Conv2d/ConvNN":
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Conv2d_ConvNN_Branching(
                                in_channels=self.in_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                kernel_size=self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                            )
                        )
                    elif self.sampling == "Spatial":
                        layers.append(
                            Conv2d_ConvNN_Spatial_Branching(
                                in_channels=self.in_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                kernel_size=self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                            )
                        )
                elif args.model == "Conv2d/ConvNN_Attn": 
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Conv2d_ConvNN_Attn_Branching(
                                in_channels=self.in_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                kernel_size=self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                                image_size=self.img_size[1:]
                            )
                        )
                    elif self.sampling == "Spatial":
                        layers.append(
                            Conv2d_ConvNN_Attn_Spatial_Branching(
                                in_channels=self.in_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                kernel_size=self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                                image_size=self.img_size[1:],
                            )
                        )
                elif args.model == "Attention/ConvNN":
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Attention_ConvNN_Branching(
                                in_channels=self.in_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                num_heads=self.num_heads,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                            )
                        )
                    elif self.sampling == "Spatial":
                        layers.append(
                            Attention_ConvNN_Spatial_Branching(
                                in_channels=self.in_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                num_heads=self.num_heads,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                            )
                        )
                elif args.model == "Attention/ConvNN_Attn":
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Attention_ConvNN_Attn_Branching( 
                                in_channels=self.in_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                num_heads=self.num_heads,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                                image_size=self.img_size[1:]
                            )
                        )
                    elif self.sampling == "Spatial":
                        layers.append(
                            Attention_ConvNN_Attn_Spatial_Branching( 
                                in_channels=self.in_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                num_heads=self.num_heads,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                                image_size=self.img_size[1:],
                            )
                        )
                elif args.model == "Conv2d/Attention":
                    layers.append(
                        Attention_Conv2d_Branching(
                            in_channels=self.in_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            kernel_size=self.k_kernel,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels,
                        )
                    )
            else: 
                if args.model == "Conv2d":
                    layers.append(
                        nn.Conv2d(
                            self.mid_ch, 
                            self.mid_ch, 
                            kernel_size=self.k_kernel, 
                            stride=1, 
                            padding=(self.k_kernel - 1) // 2 if self.k_kernel % 2 == 1 else self.k_kernel // 2
                        )
                    )
                elif args.model == "ConvNN": 
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Conv2d_NN(
                                in_channels=self.mid_ch,
                                out_channels=self.mid_ch, 
                                K=self.k_kernel,
                                stride=self.k_kernel, 
                                padding=0, 
                                shuffle_pattern=self.shuffle_pattern, 
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type,
                                location_channels=self.location_channels
                            )
                        )
                    elif self.sampling == "Spatial":
                        layers.append(
                            Conv2d_NN_Spatial(
                                in_channels=self.mid_ch,
                                out_channels=self.mid_ch, 
                                K=self.k_kernel,
                                stride=self.k_kernel, 
                                padding=0, 
                                shuffle_pattern=self.shuffle_pattern, 
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                sample_padding=0,
                                magnitude_type=self.magnitude_type,
                                location_channels=self.location_channels
                            )
                        )
                elif args.model == "ConvNN_Attn":
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Conv2d_NN_Attn(
                                in_channels=self.mid_ch,
                                out_channels=self.mid_ch, 
                                K=self.k_kernel,
                                stride=self.k_kernel, 
                                padding=0, 
                                shuffle_pattern=self.shuffle_pattern, 
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type,
                                location_channels=self.location_channels, 
                                image_size=self.img_size[1:]
                            )
                        )   
                    elif self.sampling == "Spatial":
                        layers.append(
                            Conv2d_NN_Attn_Spatial(
                                in_channels=self.mid_ch,
                                out_channels=self.mid_ch, 
                                K=self.k_kernel,
                                stride=self.k_kernel, 
                                padding=0, 
                                shuffle_pattern=self.shuffle_pattern, 
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type,
                                location_channels=self.location_channels, 
                                image_size=self.img_size[1:],
                            )   
                        )
                elif args.model == "Attention":
                    layers.append(
                        Attention2d(
                            in_channels=self.mid_ch,
                            out_channels=self.mid_ch,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels,
                        )
                    )
                elif args.model == "Conv2d/ConvNN":
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Conv2d_ConvNN_Branching(
                                in_channels = self.mid_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                kernel_size=self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                            )
                        )
                    elif self.sampling == "Spatial":
                        layers.append(
                            Conv2d_ConvNN_Spatial_Branching(
                                in_channels = self.mid_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                            )
                        )
                elif args.model == "Conv2d/ConvNN_Attn": 
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Conv2d_ConvNN_Attn_Branching(
                                in_channels = self.mid_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                kernel_size=self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                                image_size=self.img_size[1:]
                            )
                        )
                    elif self.sampling == "Spatial":
                        layers.append(
                            Conv2d_ConvNN_Attn_Spatial_Branching(
                                in_channels = self.mid_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                                image_size=self.img_size[1:],
                            )
                        )
                elif args.model == "Attention/ConvNN":
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Attention_ConvNN_Branching(
                                in_channels = self.mid_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                num_heads=self.num_heads,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                            )
                        )
                    elif self.sampling == "Spatial":
                        layers.append(
                            Attention_ConvNN_Spatial_Branching(
                                in_channels = self.mid_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                num_heads=self.num_heads,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                            )
                        )
                elif args.model == "Attention/ConvNN_Attn":
                    if self.sampling == "All" or "Random": 
                        layers.append(
                            Attention_ConvNN_Attn_Branching( 
                                in_channels = self.mid_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                num_heads=self.num_heads,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                                image_size=self.img_size[1:]
                            )
                        )
                    elif self.sampling == "Spatial":
                        layers.append(
                            Attention_ConvNN_Attn_Spatial_Branching( 
                                in_channels = self.mid_ch, 
                                out_channels = self.mid_ch,
                                channel_ratio=(self.mid_ch, self.mid_ch),
                                K = self.k_kernel,
                                shuffle_pattern=self.shuffle_pattern,
                                shuffle_scale=self.shuffle_scale,
                                num_heads=self.num_heads,
                                samples=self.num_samples,
                                magnitude_type=self.magnitude_type, 
                                location_channels=self.location_channels,
                                image_size=self.img_size[1:],
                            )
                        )
                elif args.model == "Conv2d/Attention":
                    layers.append(
                        Attention_Conv2d_Branching(
                            in_channels = self.mid_ch, 
                            out_channels = self.mid_ch,
                            channel_ratio=(self.mid_ch, self.mid_ch),
                            kernel_size=self.k_kernel,
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads,
                            location_channels=self.location_channels,
                        )
                    )
                    
                
            
            layers.append(nn.ReLU())
        
        self.features = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        
        flattened_size = self.mid_ch * self.img_size[1] * self.img_size[2]
    
        # Adjusted classifier size
        self.classifier = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(flattened_size, self.num_classes)
        )
        
        self.to(self.device)
        self.name = f"{self.model} - Sampling: {self.sampling} - K: {self.k_kernel} - N: {self.num_samples}"
        
    def forward(self, x): 
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def summary(self): 
        # Ensure the model is on CPU for torchsummary if it causes issues with MPS
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
        
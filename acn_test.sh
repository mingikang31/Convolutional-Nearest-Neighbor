#!/bin/bash

# 1. Regular Conv2d
python allconvnet_main.py --layer Conv2d --kernel_size 3 --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/Conv2d

# 2. ConvNN All Samples
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNN_All

# 3. ConvNN Random Samples
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type random --num_samples 64 --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNN_Random

# 4. ConvNN Spatial Samples
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type spatial --num_samples 8 --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNN_Spatial

# 5. ConvNN_Attn All Samples
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNNAttention_All

# 6. ConvNN_Attn Random Samples
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type random --num_samples 64 --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNNAttention_Random

# 7. ConvNN_Attn Spatial Samples
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type spatial --num_samples 8 --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNNAttention_Spatial

### Branching Networks

# 8. Branching Conv2d + ConvNN All
python allconvnet_main.py --layer Conv2d/ConvNN --num_layers 3 --K 9 --kernel_size 3 --channels 8 16 32 --sampling_type all --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/Conv2d_ConvNN_All

# 9. Branching Conv2d + ConvNN Random
python allconvnet_main.py --layer Conv2d/ConvNN --num_layers 3 --K 9 --kernel_size 3 --channels 8 16 32 --sampling_type random --num_samples 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Conv2d_ConvNN_Random

# 10. Branching Conv2d + ConvNN Spatial
python allconvnet_main.py --layer Conv2d/ConvNN --num_layers 3 --K 9 --kernel_size 3 --channels 8 16 32 --sampling_type spatial --num_samples 8 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Conv2d_ConvNN_Spatial

# 11. Branching Conv2d + ConvNN_Attn All
python allconvnet_main.py --layer Conv2d/ConvNN_Attn --num_layers 3 --K 9 --kernel_size 3 --channels 8 16 32 --sampling_type all --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Conv2d_ConvNNAttention_All

# 12. Branching Conv2d + ConvNN_Attn Random
python allconvnet_main.py --layer Conv2d/ConvNN_Attn --num_layers 3 --K 9 --kernel_size 3 --channels 8 16 32 --sampling_type random --num_samples 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Conv2d_ConvNNAttention_Random

# 13. Branching Conv2d + ConvNN_Attn Spatial
python allconvnet_main.py --layer Conv2d/ConvNN_Attn --num_layers 3 --K 9 --kernel_size 3 --channels 8 16 32 --sampling_type spatial --num_samples 8 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Conv2d_ConvNNAttention_Spatial

# 14. Branching Attention + ConvNN All
python allconvnet_main.py --layer Attention/ConvNN --num_layers 3 --K 9 --channels 8 16 32 --num_heads 4 --sampling_type all --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Attention_ConvNN_All

# 15. Branching Attention + ConvNN Random
python allconvnet_main.py --layer Attention/ConvNN --num_layers 3 --K 9 --channels 8 16 32 --num_heads 4 --sampling_type random --num_samples 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Attention_ConvNN_Random

# 16. Branching Attention + ConvNN Spatial
python allconvnet_main.py --layer Attention/ConvNN --num_layers 3 --K 9 --channels 8 16 32 --num_heads 4 --sampling_type spatial --num_samples 8 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Attention_ConvNN_Spatial

# 17. Branching Attention + ConvNN_Attn All
python allconvnet_main.py --layer Attention/ConvNN_Attn --num_layers 3 --K 9 --num_heads 4 --channels 8 16 32 --sampling_type all --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Attention_ConvNNAttention_All

# 18. Branching Attention + ConvNN_Attn Random
python allconvnet_main.py --layer Attention/ConvNN_Attn --num_layers 3 --K 9 --num_heads 4 --channels 8 16 32 --sampling_type random --num_samples 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Attention_ConvNNAttention_Random

# 19. Branching Attention + ConvNN_Attn Spatial
python allconvnet_main.py --layer Attention/ConvNN_Attn --num_layers 3 --K 9 --num_heads 4 --channels 8 16 32 --sampling_type spatial --num_samples 8 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Attention_ConvNNAttention_Spatial

# 20. Attention + Conv2d
python allconvnet_main.py --layer Conv2d/Attention --num_layers 3 --kernel_size 3 --num_heads 4 --channels 8 16 32 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/B_Attention_Conv2d

# 21. Attention
python allconvnet_main.py --layer Attention --num_layers 3 --num_heads 4 --channels 8 16 32 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACN/Attention

### Coordinate Channels
# 22. ConvNN All Samples with Coordinate Encoding
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNN_All_CoordEncoding --coordinate_encoding

# 23. ConvNN Random Samples with Coordinate Encoding
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type random --num_samples 64 --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNN_Random_CoordEncoding --coordinate_encoding

# 24. ConvNN Spatial Samples with Coordinate Encoding
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type spatial --num_samples 8 --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNN_Spatial_CoordEncoding --coordinate_encoding

# 25. ConvNN_Attn All Samples with Coordinate Encoding
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNNAttention_All_CoordEncoding --coordinate_encoding

# 26. ConvNN_Attn Random Samples with Coordinate Encoding
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type random --num_samples 64 --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNNAttention_Random_CoordEncoding --coordinate_encoding

# 27. ConvNN_Attn Spatial Samples with Coordinate Encoding
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type spatial --num_samples 8 --num_layers 3 --channels 16 32 64 --dataset cifar10 --use_amp --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/ACM/ConvNNAttention_Spatial_CoordEncoding --coordinate_encoding

### DONE with all experiments
echo "All experiments finished."
echo "Results are saved in the Output/ACM and Output/ACN directories."

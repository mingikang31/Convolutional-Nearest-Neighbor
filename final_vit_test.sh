#!/bin/bash

# ViT-Tiny-Tiny-Tiny Configuration:
# num_layers: 4
# d_hidden: 48
# d_mlp: 192
# num_heads: 3

# ### Baseline Models ###
# # Attention
# python vit_main.py --layer Attention --patch_size 16 --num_layers 4 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/Attention

# # Conv1d
# python vit_main.py --layer Conv1d --patch_size 16 --num_layers 4 --K 3 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/Conv1d

# # Conv1d Attention
# python vit_main.py --layer Conv1dAttention --patch_size 16 --num_layers 4 --K 3 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/Conv1dAttention

# # KVT Attention
# python vit_main.py --layer KvtAttention --patch_size 16 --num_layers 4 --K 3 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/KvtAttention

# ConvNN All
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNN_All_Decay_01 --weight_decay 0.1

# ConvNN Random
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNN_Random_Decay_01 --weight_decay 0.1

# ConvNN Spatial
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNN_Spatial_Decay_01 --weight_decay 0.1

# ConvNNAttention All
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNNAttention_All_Decay_01 --weight_decay 0.1

# ConvNNAttention Random
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNNAttention_Random_Decay_01 --weight_decay 0.1

# ConvNNAttention Spatial
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNNAttention_Spatial_Decay_01 --weight_decay 0.1

# ConvNN All Coord
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNN_All_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# ConvNN Random Coord
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNN_Random_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# ConvNN Spatial Coord
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNN_Spatial_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# ConvNNAttention All Coord
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNNAttention_All_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# ConvNNAttention Random Coord
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNNAttention_Random_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# ConvNNAttention Spatial Coord
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/ConvNNAttention_Spatial_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# # Local Attention
# python vit_main.py --layer LocalAttention --patch_size 16 --num_layers 4 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/LocalAttention 

# # NeighborhoodAttention
# python vit_main.py --layer NeighborhoodAttention --patch_size 16 --num_layers 4 --K 3 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR10/NeighborhoodAttention



# Attention
python vit_main.py --layer Attention --patch_size 16 --num_layers 4 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/Attention

# Conv1d
python vit_main.py --layer Conv1d --patch_size 16 --num_layers 4 --K 3 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/Conv1d

# Conv1d Attention
python vit_main.py --layer Conv1dAttention --patch_size 16 --num_layers 4 --K 3 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/Conv1dAttention

# KVT Attention
python vit_main.py --layer KvtAttention --patch_size 16 --num_layers 4 --K 3 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/KvtAttention

# ConvNN All
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_All

# ConvNN Random
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_Random

# ConvNN Spatial
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_Spatial

# ConvNNAttention All
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_All

# ConvNNAttention Random
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_Random

# ConvNNAttention Spatial
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_Spatial

# ConvNN All Coord
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_All_Coord --coordinate_encoding

# ConvNN Random Coord
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_Random_Coord --coordinate_encoding

# ConvNN Spatial Coord
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_Spatial_Coord --coordinate_encoding

# ConvNNAttention All Coord
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_All_Coord --coordinate_encoding

# ConvNNAttention Random Coord
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_Random_Coord --coordinate_encoding

# ConvNNAttention Spatial Coord
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_Spatial_Coord --coordinate_encoding

# Local Attention
python vit_main.py --layer LocalAttention --patch_size 16 --num_layers 4 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/LocalAttention

# # NeighborhoodAttention
# python vit_main.py --layer NeighborhoodAttention --patch_size 16 --num_layers 4 --K 3 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/NeighborhoodAttention








# ConvNN All
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_All_Decay_01 --weight_decay 0.1

# ConvNN Random
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_Random_Decay_01 --weight_decay 0.1

# ConvNN Spatial
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_Spatial_Decay_01 --weight_decay 0.1

# ConvNNAttention All
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_All_Decay_01 --weight_decay 0.1

# ConvNNAttention Random
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_Random_Decay_01 --weight_decay 0.1

# ConvNNAttention Spatial
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_Spatial_Decay_01 --weight_decay 0.1

# ConvNN All Coord
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_All_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# ConvNN Random Coord
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_Random_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# ConvNN Spatial Coord
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNN_Spatial_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# ConvNNAttention All Coord
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type all --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_All_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# ConvNNAttention Random Coord
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_Random_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1

# ConvNNAttention Spatial Coord
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 4 --K 3 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 48 --d_mlp 192 --dropout 0.1 --attention_dropout 0.1 --dataset cifar100 --num_epochs 50 --seed 0 --output_dir ./Output/Final_results/ViT-Tiny-Tiny-Tiny/CIFAR100/ConvNNAttention_Spatial_Coord_Decay_01 --coordinate_encoding --weight_decay 0.1
#!/bin/bash

# ViT-Tiny Configuration:
# num_layers: 12
# d_hidden: 192
# d_mlp: 768
# num_heads: 3

# Attention
python vit_main.py --layer Attention --patch_size 16 --num_layers 12 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/ViT-Tiny/Attention

# Conv1d
python vit_main.py --layer Conv1d --patch_size 16 --num_layers 12 --K 9 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/ViT-Tiny/Conv1d

# Conv1d Attention
python vit_main.py --layer Conv1dAttention --patch_size 16 --num_layers 12 --K 9 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/ViT-Tiny/Conv1dAttention

# KVT Attention
python vit_main.py --layer KvtAttention --patch_size 16 --num_layers 12 --K 9 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/VIT-Tiny/KvtAttention

# ConvNN All
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/VIT-Tiny/ConvNN_All

# ConvNN Random
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/VIT-Tiny/ConvNN_Random

# ConvNN Spatial
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/ViT-Tiny/ConvNN_Spatial

# ConvNNAttention All
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 12 --K 9 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/ViT-Tiny/ConvNNAttention_All

# ConvNNAttention Random
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 12 --K 9 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/ViT-Tiny/ConvNNAttention_Random

# ConvNNAttention Spatial
python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 12 --K 9 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/ViT-Tiny/ConvNNAttention_Spatial

# Local Attention
python vit_main.py --layer LocalAttention --patch_size 16 --num_layers 12 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/ViT-Tiny/LocalAttention


# NeighborhoodAttention
python vit_main.py --layer NeighborhoodAttention --patch_size 16 --num_layers 12 --K 9 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --use_amp --seed 0 --output_dir ./Output/ViT-Tiny/NeighborhoodAttention

echo "All experiments finished."

#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=CVPR2
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi



## VGG11 + VGG13 
## I. CIFAR10 
# 1. Baseline 
# SGD 


python main.py --model vgg11 --layer Branching_Attn --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.500 --criterion CrossEntropy --batch_size 128 --num_epochs 100 --optimizer sgd --momentum 0.9 --weight_decay 5e-4 --lr 5e-2 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/TEST-Oct24-Jupyter/VGG11/BranchingConvNNAttn_K9_col_col_br0500_s42 --clip_grad_norm 1.0

python main.py --model vgg11 --layer Branching_Attn --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 1.000 --criterion CrossEntropy --batch_size 128 --num_epochs 100 --optimizer sgd --momentum 0.9 --weight_decay 5e-4 --lr 5e-2 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/TEST-Oct24-Jupyter/VGG11/BranchingConvNNAttn_K9_col_col_br1000_s42 --clip_grad_norm 1.0


python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.500 --criterion CrossEntropy --batch_size 128 --num_epochs 100 --optimizer sgd --momentum 0.9 --weight_decay 5e-4 --lr 5e-2 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/TEST-Oct24-Jupyter/VGG11/BranchingConvNN_K9_col_col_br0500_s42 --clip_grad_norm 1.0

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 1.000 --criterion CrossEntropy --batch_size 128 --num_epochs 100 --optimizer sgd --momentum 0.9 --weight_decay 5e-4 --lr 5e-2 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/TEST-Oct24-Jupyter/VGG11/BranchingConvNN_K9_col_col_br1000_s42 --clip_grad_norm 1.0


# Baseline
python main.py --model vgg11 --layer Branching_Attn --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.000 --criterion CrossEntropy --batch_size 128 --num_epochs 100 --optimizer sgd --momentum 0.9 --weight_decay 5e-4 --lr 5e-2 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/TEST-Oct24-Jupyter/VGG11/BranchingConvNNAttn_K9_col_col_br0000_s42 --clip_grad_norm 1.0


python main.py --model vgg11 --layer Conv2d --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.000 --criterion CrossEntropy --batch_size 128 --num_epochs 100 --optimizer sgd --momentum 0.9 --weight_decay 5e-4 --lr 5e-2 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/TEST-Oct24-Jupyter/VGG11/Conv2d_K3_s42 --clip_grad_norm 1.0
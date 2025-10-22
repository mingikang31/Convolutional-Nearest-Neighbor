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

## Need to change the data augmentation for generalization experiments

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.000 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-4 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/TEST-Oct22-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br0000_s42_relu


python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.500 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-4 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/TEST-Oct22-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br0500_s42_relu

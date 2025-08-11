#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --job-name=vgg-exps
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu



cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

### CIFAR10 Experiments

# VGG A - Conv2d 
python main.py \
    --epochs 100 \ 

# VGG A - ConvNN All Samples 

# VGG A - ConvNN Random Samples 

# VGG A - ConvNN Spatial Samples 


# VGG A - ConvNN_Attn All Samples

# VGG A - ConvNN_Attn Random Samples 

# VGG A - ConvNN_Attn Spatial Samples

### CIFAR100 Experiments

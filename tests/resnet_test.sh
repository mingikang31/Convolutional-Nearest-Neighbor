#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:rtx5090:1
#SBATCH --job-name=resnet-exps
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu


cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

### CIFAR10 Experiments

# ResNet18 - Conv2d 
python main.py \
    --epochs 100 \ 

# ResNet18 - ConvNN All Samples 

# ResNet18 - ConvNN Random Samples 

# ResNet18 - ConvNN Spatial Samples 


# ResNet18 - ConvNN_Attn All Samples

# ResNet18 - ConvNN_Attn Random Samples 

# ResNet18 - ConvNN_Attn Spatial Samples

### CIFAR100 Experiments

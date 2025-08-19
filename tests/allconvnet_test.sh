#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:rtx3080:1
#SBATCH --job-name=allconvnet-exps
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi

### CIFAR10 Experiments

# AllConvNet - Conv2d 
python main.py 

# AllConvNet - ConvNN All Samples 

# AllConvNet - ConvNN Random Samples 

# AllConvNet - ConvNN Spatial Samples 


# AllConvNet - ConvNN_Attn All Samples

# AllConvNet - ConvNN_Attn Random Samples 

# AllConvNet - ConvNN_Attn Spatial Samples

### CIFAR100 Experiments



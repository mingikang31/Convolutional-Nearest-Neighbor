#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=Sanity_Test
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi

# ConvNN
python vgg_main.py --layer ConvNN --K 9 --coordinate_encoding --sampling_type all --num_epochs 50 --output_dir ./Output/Sep_8/vgg/Distance/1e_3/ConvNN_All_K9_0_p1 --padding 1 --seed 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-3


# Conv2d
# python allconvnet_main.py --layer Conv2d --num_layers 3 --channels 32 16 8 --kernel_size 3 --num_epochs 150 --output_dir ./Output/Sep_6_no_bias/Color/1e_3/Conv2d_K3 --lr_step 2 --lr_gamma 0.95 --lr 1e-3

# python allconvnet_main.py --layer Conv2d --num_layers 3 --channels 32 16 8 --kernel_size 2 --num_epochs 150 --output_dir ./Output/Sep_6_no_bias/Color/1e_3/Conv2d_K2 --lr_step 2 --lr_gamma 0.95 --lr 1e-3

# python allconvnet_main.py --layer Conv2d --num_layers 3 --channels 32 16 8 --kernel_size 1 --num_epochs 150 --output_dir ./Output/Sep_6_no_bias/Color/1e_3/Conv2d_K1 --lr_step 2 --lr_gamma 0.95 --lr 1e-3

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

# python allconvnet_main.py --layer Conv2d --num_layers 3 --channels 32 16 8 --kernel_size 3 --num_epochs 75 --output_dir ./Output/Sep_1_Sanity/Conv2d_K3

# python allconvnet_main.py --layer Conv2d --num_layers 3 --channels 32 16 8 --kernel_size 2 --num_epochs 75 --output_dir ./Output/Sep_1_Sanity/Conv2d_K2

# python allconvnet_main.py --layer Conv2d --num_layers 3 --channels 32 16 8 --kernel_size 1 --num_epochs 75 --output_dir ./Output/Sep_1_Sanity/Conv2d_K1

### DIST
## SEED 0 

python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 9 --coordinate_encoding --sampling_type all --num_epochs 75 --output_dir ./Output/Sep_2_Sanity_Dist/ConvNN_All_K9_0_p0 --padding 0 --seed 0

python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 4 --coordinate_encoding --sampling_type all --num_epochs 75 --output_dir ./Output/Sep_2_Sanity_Dist/ConvNN_All_K4_0 --padding 0 --seed 0

python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 1 --coordinate_encoding --sampling_type all --num_epochs 75 --output_dir ./Output/Sep_2_Sanity_Dist/ConvNN_All_K1_0 --padding 0 --seed 0


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

# python allconvnet_main.py --layer Conv2d --num_layers 3 --channels 32 16 8 --kernel_size 3 --num_epochs 75 --output_dir ./Output/Aug_30_Sanity/Conv2d_K3

python allconvnet_main.py --layer Conv2d --num_layers 3 --channels 32 16 8 --kernel_size 2 --num_epochs 75 --output_dir ./Output/Aug_30_Sanity/Conv2d_K2


# python allconvnet_main.py --layer Conv2d --num_layers 3 --channels 32 16 8 --kernel_size 1 --num_epochs 75 --output_dir ./Output/Aug_30_Sanity/Conv2d_K1

python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 9 --sampling_type all --num_epochs 75 --coordinate_encoding --output_dir ./Output/Aug_30_Sanity/ConvNN_All_K9_0 --padding 1 --seed 0

python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 4 --sampling_type all --num_epochs 75 --coordinate_encoding --output_dir ./Output/Aug_30_Sanity/ConvNN_All_K4_0 --padding 1 --seed 0

python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 1 --sampling_type all --num_epochs 75 --coordinate_encoding --output_dir ./Output/Aug_30_Sanity/ConvNN_All_K1_0 --padding 0 --seed 0


python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 9 --sampling_type all --num_epochs 75 --coordinate_encoding --output_dir ./Output/Aug_30_Sanity/ConvNN_All_K9_42 --padding 1 --seed 42

python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 4 --sampling_type all --num_epochs 75 --coordinate_encoding --output_dir ./Output/Aug_30_Sanity/ConvNN_All_K4_42 --padding 1 --seed 42

python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 1 --sampling_type all --num_epochs 75 --coordinate_encoding --output_dir ./Output/Aug_30_Sanity/ConvNN_All_K1_42 --padding 0 --seed 42


python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 9 --sampling_type all --num_epochs 75 --coordinate_encoding --output_dir ./Output/Aug_30_Sanity/ConvNN_All_K9_102 --padding 1 --seed 102

python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 4 --sampling_type all --num_epochs 75 --coordinate_encoding --output_dir ./Output/Aug_30_Sanity/ConvNN_All_K4_102 --padding 1 --seed 102

python allconvnet_main.py --layer ConvNN --num_layers 3 --channels 32 16 8 --K 1 --sampling_type all --num_epochs 75 --coordinate_encoding --output_dir ./Output/Aug_30_Sanity/ConvNN_All_K1_102 --padding 0 --seed 102

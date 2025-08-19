#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=default-exps
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi
### CIFAR10 Experiments



### ResNet Experiments 
python resnet_main.py --layer Conv2d --num_layers 3 --channels 8 16 32 --kernel_size 3 --num_epoch 5 --output_dir ./Output/TEST/ResNet-RTX3080/Conv2d

python resnet_main.py --layer Conv2d_New --num_layers 3 --channels 8 16 32 --kernel_size 3 --shuffle_pattern BA --shuffle_scale 2 --num_epoch 5 --output_dir ./Output/TEST/ResNet-RTX3080/Conv2d_New

python resnet_main.py --layer Conv2d_New_1d --num_layers 3 --channels 8 16 32 --K 9 --shuffle_pattern BA --shuffle_scale 2 --num_epoch 5 --output_dir ./Output/TEST/ResNet-RTX3080/Conv2d_New_1d

python resnet_main.py --layer ConvNN --num_layers 3 --channels 8 16 32 --K 9 --sampling_type all --shuffle_pattern BA --shuffle_scale 2 --num_epoch 5 --output_dir ./Output/TEST/ResNet-RTX3080/ConvNN_All

python resnet_main.py --layer ConvNN --num_layers 3 --channels 8 16 32 --K 9 --sampling_type random --num_samples 64 --shuffle_pattern BA --shuffle_scale 2 --num_epoch 5 --output_dir ./Output/TEST/ResNet-RTX3080/ConvNN_Random

python resnet_main.py --layer ConvNN --num_layers 3 --channels 8 16 32 --K 9 --sampling_type spatial --num_samples 8 --shuffle_pattern BA --shuffle_scale 2 --num_epoch 5 --output_dir ./Output/TEST/ResNet-RTX3080/ConvNN_Spatial

python resnet_main.py --layer ConvNN_Attn --num_layers 3 --channels 8 16 32 --K 9 --sampling_type all --shuffle_pattern BA --shuffle_scale 2 --num_epoch 5 --output_dir ./Output/TEST/ResNet-RTX3080/ConvNN_All_Attn

python resnet_main.py --layer ConvNN_Attn --num_layers 3 --channels 8 16 32 --K 9 --sampling_type random --num_samples 64 --shuffle_pattern BA --shuffle_scale 2 --num_epoch 5 --output_dir ./Output/TEST/ResNet-RTX3080/ConvNN_Random_Attn

python resnet_main.py --layer ConvNN_Attn --num_layers 3 --channels 8 16 32 --K 9 --sampling_type spatial --num_samples 8 --shuffle_pattern BA --shuffle_scale 2 --num_epoch 5 --output_dir ./Output/TEST/ResNet-RTX3080/ConvNN_Spatial_Attn

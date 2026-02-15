#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=480G
#SBATCH -p mixed --gres=gpu:pro6000:1
#SBATCH --cpus-per-gpu=80
#SBATCH --job-name=convnn-imgnet
#SBATCH --time=720:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-pro6000

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor 

python main.py \
    --model resnet50 \
    --layer Branching \
    --K 9 \
    --kernel_size 3 \
    --padding 1 \
    --sampling_type all \
    --num_samples -1 \
    --dataset imagenet1k \
    --compile \
    --use_amp \
    --num_workers 12 \
    --pin_memory \
    --batch_size 512 \
    --data_path /mnt/research/j.farias/mkang2/Datasets \
    --output_dir ./ImageNet-Output/ResNet-50/Branching_All_K9_Ks3
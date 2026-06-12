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
    --layer Conv2d \
    --kernel_size 2 \
    --dataset imagenet1k \
    --compile \
    --use_amp \
    --num_workers 12 \
    --pin_memory \
    --batch_size 512 \
    --data_path /mnt/research/j.farias/mkang2/Datasets \
    --output_dir ./ImageNet-Output/ResNet-50/Convolution_Ks2/

python main.py \
    --model resnet50 \
    --layer Conv2d \
    --kernel_size 3 \
    --dataset imagenet1k \
    --compile \
    --use_amp \
    --num_workers 12 \
    --pin_memory \
    --batch_size 512 \
    --data_path /mnt/research/j.farias/mkang2/Datasets \
    --output_dir ./ImageNet-Output/ResNet-50/Convolution_Ks3/

python main.py \
    --model resnet50 \
    --layer Conv2d \
    --kernel_size 4 \
    --dataset imagenet1k \
    --compile \
    --use_amp \
    --num_workers 12 \
    --pin_memory \
    --batch_size 512 \
    --data_path /mnt/research/j.farias/mkang2/Datasets \
    --output_dir ./ImageNet-Output/ResNet-50/Convolution_Ks4/

python main.py \
    --model resnet50 \
    --layer Conv2d \
    --kernel_size 5 \
    --dataset imagenet1k \
    --compile \
    --use_amp \
    --num_workers 12 \
    --pin_memory \
    --batch_size 512 \
    --data_path /mnt/research/j.farias/mkang2/Datasets \
    --output_dir ./ImageNet-Output/ResNet-50/Convolution_Ks5/


# python main.py \
#     --model resnet50 \
#     --layer Branching \
#     --K 4 \
#     --kernel_size 3 \
#     --padding 1 \
#     --sampling_type all \
#     --num_samples -1 \
#     --dataset imagenet1k \
#     --compile \
#     --use_amp \
#     --num_workers 12 \
#     --pin_memory \
#     --batch_size 512 \
#     --data_path /mnt/research/j.farias/mkang2/Datasets \
#     --output_dir ./ImageNet-Output/ResNet-50/Branching_All_K6_Ks3/


# python main.py \
#     --model resnet50 \
#     --layer Branching \
#     --K 16 \
#     --kernel_size 3 \
#     --padding 1 \
#     --sampling_type random \
#     --num_samples 32 \
#     --dataset imagenet1k \
#     --compile \
#     --use_amp \
#     --num_workers 12 \
#     --pin_memory \
#     --batch_size 512 \
#     --data_path /mnt/research/j.farias/mkang2/Datasets \
#     --output_dir ./ImageNet-Output/ResNet-50/Branching_Rand_K16_Ks3_N32/

# python main.py \
#     --model resnet50 \
#     --layer Branching \
#     --K 16 \
#     --kernel_size 3 \
#     --padding 1 \
#     --sampling_type random \
#     --num_samples 48 \
#     --dataset imagenet1k \
#     --compile \
#     --use_amp \
#     --num_workers 12 \
#     --pin_memory \
#     --batch_size 512 \
#     --data_path /mnt/research/j.farias/mkang2/Datasets \
#     --output_dir ./ImageNet-Output/ResNet-50/Branching_Rand_K16_Ks3_N48/


# python main.py \
#     --model resnet50 \
#     --layer Branching \
#     --K 16 \
#     --kernel_size 3 \
#     --padding 1 \
#     --sampling_type random \
#     --num_samples 64 \
#     --dataset imagenet1k \
#     --compile \
#     --use_amp \
#     --num_workers 12 \
#     --pin_memory \
#     --batch_size 512 \
#     --data_path /mnt/research/j.farias/mkang2/Datasets \
#     --output_dir ./ImageNet-Output/ResNet-50/Branching_Rand_K16_Ks3_N64/


python main.py \
    --model resnet50 \
    --layer Branching \
    --K 4 \
    --kernel_size 2 \
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
    --output_dir ./ImageNet-Output/ResNet-50/Branching_All_K4_Ks2/


python main.py \
    --model resnet50 \
    --layer Branching \
    --K 16 \
    --kernel_size 4 \
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
    --output_dir ./ImageNet-Output/ResNet-50/Branching_All_K16_Ks4/

python main.py \
    --model resnet50 \
    --layer Branching \
    --K 25 \
    --kernel_size 5 \
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
    --output_dir ./ImageNet-Output/ResNet-50/Branching_All_K25_Ks5/
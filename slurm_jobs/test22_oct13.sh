#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=Sep29_NTest
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi
#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=Sep29_NTest
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi

# Conv2d ResNet 


python main.py --model resnet18 --layer Conv2d --kernel_size 3 --num_epochs 60 --output_dir ./Output/Oct13_ResNet/resnet18_1e-5/CIFAR10/Conv2d_K3_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --dataset cifar10 --weight_decay 5e-4


python main.py --model resnet18 --layer ConvNN --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/resnet18_1e-5/CIFAR10/ConvNN_K9_Col_Col_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --dataset cifar10 --weight_decay 5e-4 --similarity_type Col --aggregation_type Col --magnitude_type cosine


## ConvNN Attention 
python main.py --model resnet18 --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Oct13_ResNet/resnet18_1e-5/CIFAR10/ConvBranch_KS3_K9_Col_Col_r0500_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --dataset cifar10 --weight_decay 5e-4 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.500

python main.py --model resnet18 --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Oct13_ResNet/resnet18_1e-5/CIFAR10/ConvBranch_KS3_K9_LocCol_LocCol_r0500_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --dataset cifar10 --weight_decay 5e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.500
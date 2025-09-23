#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=PT4_Exp_VGG
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi


# Conv2d
python vgg_main.py --layer Conv2d --kernel_size 3 --num_epochs 60 --output_dir ./Output/Sep_22_Branching/vgg_1e-5/CIFAR10/Conv2d_K3_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5

## Branching
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_22_Branching/vgg_1e-5_cos/CIFAR10/Col_Col_Branch/ConvBranch_K9_42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine # Done

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_22_Branching/vgg_1e-5_cos/CIFAR10/LocCol_Col_Branch/ConvBranch_K9_42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --lambda_param 0.5 # Done

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_22_Branching/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_K9_42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --lambda_param 0.5 # Done


# ConvNN
python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_22_Branching/vgg_1e-5_cos/CIFAR10/LocCol_Col/ConvNN_K9_42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --lambda_param 0.5 # Done

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_22_Branching/vgg_1e-5_cos/CIFAR10/LocCol_LocCol/ConvNN_K9_42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --lambda_param 0.5 # Done

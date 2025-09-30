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

# N Test for Random and Spatial Sampling 
# N = 16, 32, 64, 128, 256, etc.


# CIFAR10 

# N = 16 (4x4)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 16 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand16_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 4 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat4_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

# N = 64 (8x8)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 64 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand64_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 8 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat8_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

# N = 144 (12x12)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 144 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand144_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 12 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat12_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

# N = 256 (16x16)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 256 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand256_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 16 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat16_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

# N = 400 (20x20)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 400 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand400_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 20 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat20_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

# N = 576 (24x24)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 576 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand576_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 24 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat24_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

# N = 784 (28x28)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 784 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand784_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 28 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat28_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25





## CIFAR100


# N = 16 (4x4)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 16 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand16_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 4 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat4_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# N = 64 (8x8)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 64 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand64_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 8 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat8_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# N = 144 (12x12)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 144 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand144_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 12 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat12_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# N = 256 (16x16)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 256 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand256_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 16 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat16_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# N = 400 (20x20)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 400 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand400_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 20 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat20_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# N = 576 (24x24)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 576 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand576_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 24 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat24_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# N = 784 (28x28)
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type random --num_samples 784 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_rand784_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type spatial --num_samples 28 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NTest/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_spat28_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100
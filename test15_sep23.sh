#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=VGG_Branching_Ratio_Sep23
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi


# Conv2d
# python vgg_main.py --layer Conv2d --kernel_size 3 --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5/CIFAR10/Conv2d_K3_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5

## Branching Col, Col
# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/Col_Col_Branch/ConvBranch_K9_r000_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.0

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/Col_Col_Branch/ConvBranch_K9_r0125_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.125

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/Col_Col_Branch/ConvBranch_K9_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.25

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/Col_Col_Branch/ConvBranch_K9_r050_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.50

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/Col_Col_Branch/ConvBranch_K9_r075_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.75

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/Col_Col_Branch/ConvBranch_K9_r0875_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.875

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/Col_Col_Branch/ConvBranch_K9_r100_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 1.0


## Branching Loc_Col, Col
# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_Col_Branch/ConvBranch_K9_r000_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.0

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_Col_Branch/ConvBranch_K9_r0125_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.125

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_Col_Branch/ConvBranch_K9_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.25

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_Col_Branch/ConvBranch_K9_r050_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.50

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_Col_Branch/ConvBranch_K9_r075_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.75

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_Col_Branch/ConvBranch_K9_r0875_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.875

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_Col_Branch/ConvBranch_K9_r100_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 1.0

## Branching Loc_Col, Loc_Col
# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_K9_r000_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.0

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_K9_r0125_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.125

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_K9_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_K9_r050_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.50

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_K9_r075_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.75

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_K9_r0875_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.875

# python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_23_Branching/vgg_1e-5_cos/CIFAR10/LocCol_LocCol_Branch/ConvBranch_K9_r100_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 1.0


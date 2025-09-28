#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=VGG_Sep28
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi

#### Test 15 Sep 23
# Conv2d
python vgg_main.py --layer Conv2d --kernel_size 3 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5/CIFAR100/Conv2d_K3_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --dataset cifar100

python vgg_main.py --layer Conv2d --kernel_size 2 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5/CIFAR100/Conv2d_K2_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --dataset cifar100

python vgg_main.py --layer Conv2d --kernel_size 1 --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5/CIFAR100/Conv2d_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --dataset cifar100


## Branching Col, Col
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/Col_Col_Branch/ConvBranch_K9_r000_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.0 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/Col_Col_Branch/ConvBranch_K9_r0125_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.125 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/Col_Col_Branch/ConvBranch_K9_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/Col_Col_Branch/ConvBranch_K9_r050_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.50 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/Col_Col_Branch/ConvBranch_K9_r075_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.75 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/Col_Col_Branch/ConvBranch_K9_r0875_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.875 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/Col_Col_Branch/ConvBranch_K9_r100_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine --branch_ratio 1.0 --dataset cifar100


## Branching Loc_Col, Col
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_Col_Branch/ConvBranch_K9_r000_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.0 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_Col_Branch/ConvBranch_K9_r0125_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.125 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_Col_Branch/ConvBranch_K9_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_Col_Branch/ConvBranch_K9_r050_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.50 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_Col_Branch/ConvBranch_K9_r075_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.75 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_Col_Branch/ConvBranch_K9_r0875_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 0.875 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_Col_Branch/ConvBranch_K9_r100_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --branch_ratio 1.0 --dataset cifar100

# Branching Loc_Col, Loc_Col
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_K9_r000_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.0 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_K9_r0125_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.125 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_K9_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_K9_r050_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.50 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_K9_r075_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.75 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_K9_r0875_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.875 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit_Ks/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_K9_r100_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 1.0 --dataset cifar100

#### Test16 Sep 24

# Branching Loc_Col, Loc_Col
# # Conv2d Kernel Size Test
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_r000_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.0 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 2 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_r000_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.0 --dataset cifar100

python vgg_main.py --layer Branching --kernel_size 1 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_r000_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.0 --dataset cifar100

# # K Test with Kernel Size 3, Branch Ratio 0.25
# K = 1
python vgg_main.py --layer Branching --kernel_size 3 --K 1 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K1_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 2
python vgg_main.py --layer Branching --kernel_size 3 --K 2 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K2_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 3 
python vgg_main.py --layer Branching --kernel_size 3 --K 3 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K3_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 4 
python vgg_main.py --layer Branching --kernel_size 3 --K 4 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K4_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 5
python vgg_main.py --layer Branching --kernel_size 3 --K 5 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K5_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 6 
python vgg_main.py --layer Branching --kernel_size 3 --K 6 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K6_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 7
python vgg_main.py --layer Branching --kernel_size 3 --K 7 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K7_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 8
python vgg_main.py --layer Branching --kernel_size 3 --K 8 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K8_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 9
python vgg_main.py --layer Branching --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K9_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 10
python vgg_main.py --layer Branching --kernel_size 3 --K 10 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K10_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100


# K = 11
python vgg_main.py --layer Branching --kernel_size 3 --K 11 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K11_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100


# K = 12
python vgg_main.py --layer Branching --kernel_size 3 --K 12 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS3_K12_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100



# # K Test with Kernel Size 2, Branch Ratio 0.25
# K = 1
python vgg_main.py --layer Branching --kernel_size 2 --K 1 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K1_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 2
python vgg_main.py --layer Branching --kernel_size 2 --K 2 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K2_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 3
python vgg_main.py --layer Branching --kernel_size 2 --K 3 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K3_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 4
python vgg_main.py --layer Branching --kernel_size 2 --K 4 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K4_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 5
python vgg_main.py --layer Branching --kernel_size 2 --K 5 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K5_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 6
python vgg_main.py --layer Branching --kernel_size 2 --K 6 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K6_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 7
python vgg_main.py --layer Branching --kernel_size 2 --K 7 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K7_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 8
python vgg_main.py --layer Branching --kernel_size 2 --K 8 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K8_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 9
python vgg_main.py --layer Branching --kernel_size 2 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K9_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 10
python vgg_main.py --layer Branching --kernel_size 2 --K 10 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K10_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 11
python vgg_main.py --layer Branching --kernel_size 2 --K 11 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K11_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 12
python vgg_main.py --layer Branching --kernel_size 2 --K 12 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS2_K12_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100


# # K Test with Kernel Size 1, Branch Ratio 0.25
# K = 1
python vgg_main.py --layer Branching --kernel_size 1 --K 1 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K1_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 2
python vgg_main.py --layer Branching --kernel_size 1 --K 2 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K2_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 3 
python vgg_main.py --layer Branching --kernel_size 1 --K 3 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K3_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 4 
python vgg_main.py --layer Branching --kernel_size 1 --K 4 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K4_r025_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 5
python vgg_main.py --layer Branching --kernel_size 1 --K 5 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K5_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 6 
python vgg_main.py --layer Branching --kernel_size 1 --K 6 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K6_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 7
python vgg_main.py --layer Branching --kernel_size 1 --K 7 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K7_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 8 
python vgg_main.py --layer Branching --kernel_size 1 --K 8 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K8_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 9
python vgg_main.py --layer Branching --kernel_size 1 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K9_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100

# K = 10
python vgg_main.py --layer Branching --kernel_size 1 --K 10 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K10_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100


# K = 11
python vgg_main.py --layer Branching --kernel_size 1 --K 11 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K11_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100


# K = 12
python vgg_main.py --layer Branching --kernel_size 1 --K 12 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_29_Branching_NoSplit/vgg_1e-5_cos/CIFAR100/LocCol_LocCol_Branch/ConvBranch_KS1_K12_r025_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --branch_ratio 0.25 --dataset cifar100


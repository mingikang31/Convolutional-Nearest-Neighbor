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

### VGG Experiments
# CIFAR10 
## I. Conv2d K3, K2 
python vgg_main.py --layer Conv2d --kernel_size 3 --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5/CIFAR10/Conv2d_K3_s42 --seed 42 --lr_step 2 --lr_gamma 0.95 --lr 1e-5

python vgg_main.py --layer Conv2d --kernel_size 2 --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5/CIFAR10/Conv2d_K2_s42 --seed 42 --lr_step 2 --lr_gamma 0.95 --lr 1e-5

python vgg_main.py --layer Conv2d --kernel_size 3 --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5/CIFAR100/Conv2d_K3_s42 --seed 42 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --dataset cifar100

python vgg_main.py --layer Conv2d --kernel_size 2 --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5/CIFAR100/Conv2d_K2_s42 --seed 42 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --dataset cifar100

## II. ConvNN K1-10 All Sampling - Loc, Col euclidean similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/Loc_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/Loc_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/Loc_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --magnitude_type euclidean

## III. ConvNN K1-10 All Sampling - Loc_Col, Col euclidean similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/LocCol_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/LocCol_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/LocCol_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type euclidean


## IV. ConvNN K1-10 All Sampling - Loc_Col, Loc_Col euclidean similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/LocCol_LocCol/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/LocCol_LocCol/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/LocCol_LocCol/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type euclidean

## V. ConvNN K1-10 All Sampling - Col, Col euclidean similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/Col_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --dataset cifar10 --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/Col_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --dataset cifar10 --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR10/Col_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --dataset cifar10 --magnitude_type euclidean

# CIFAR100
## I. Conv2d K3, K2

## II. ConvNN K1-10 All Sampling - Loc, Col euclidean similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/Loc_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --dataset cifar100 --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/Loc_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --dataset cifar100 --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/Loc_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --dataset cifar100 --magnitude_type euclidean

## III. ConvNN K1-10 All Sampling - Loc_Col, Col euclidean similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/LocCol_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100 --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/LocCol_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100 --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/LocCol_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100 --magnitude_type euclidean

## IV. ConvNN K1-10 All Sampling - Loc_Col, Loc_Col euclidean similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/LocCol_LocCol/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/LocCol_LocCol/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/LocCol_LocCol/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type euclidean

## V. ConvNN K1-10 All Sampling - Col, Col euclidean similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/Col_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --dataset cifar100 --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/Col_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --dataset cifar100 --magnitude_type euclidean

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_eucl/CIFAR100/Col_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --dataset cifar100 --magnitude_type euclidean



## II. ConvNN K1-10 All Sampling - Loc, Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/Loc_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/Loc_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/Loc_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --magnitude_type cosine

## III. ConvNN K1-10 All Sampling - Loc_Col, Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/LocCol_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/LocCol_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/LocCol_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine

## IV. ConvNN K1-10 All Sampling - Loc_Col, Loc_Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/LocCol_LocCol/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/LocCol_LocCol/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/LocCol_LocCol/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine

# V. ConvNN K1-10 All Sampling - Col, Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/Col_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/Col_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/Col_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --magnitude_type cosine

# CIFAR100
## I. Conv2d K3, K2


## II. ConvNN K1-10 All Sampling - Loc, Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/Loc_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --dataset cifar100 --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/Loc_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --dataset cifar100 --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/Loc_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc --aggregation_type Col --dataset cifar100 --magnitude_type cosine

## III. ConvNN K1-10 All Sampling - Loc_Col, Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/LocCol_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100 --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/LocCol_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100 --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/LocCol_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100 --magnitude_type cosine

## IV. ConvNN K1-10 All Sampling - Loc_Col, Loc_Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/LocCol_LocCol/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/LocCol_LocCol/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/LocCol_LocCol/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type cosine


# V. ConvNN K1-10 All Sampling - Col, Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/Col_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --dataset cifar100 --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/Col_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --dataset cifar100 --magnitude_type cosine

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/Col_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Col --aggregation_type Col --dataset cifar100 --magnitude_type cosine




### Sampling - Random & Spatial Sampling 
## CIFAR10
# Random with N = 64
python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/Loc_Col/ConvNN_Rand_K4_N64_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar10 --magnitude_type cosine --sampling_type random --num_samples 64

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/Loc_Col/ConvNN_Rand_K9_N64_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar10 --magnitude_type cosine --sampling_type random --num_samples 64

# Spatial with N = 8
python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/Loc_Col/ConvNN_Spat_K4_N8_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar10 --magnitude_type cosine --sampling_type spatial --num_samples 8

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR10/Loc_Col/ConvNN_Spat_K9_N8_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar10 --magnitude_type cosine --sampling_type spatial --num_samples 8



## CIFAR100 
# Random with N = 64
python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/Loc_Col/ConvNN_Rand_K4_N64_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type cosine --sampling_type random --num_samples 64

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/Loc_Col/ConvNN_Rand_K9_N64_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type cosine --sampling_type random --num_samples 64

# Spatial with N = 8
python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/Loc_Col/ConvNN_Spat_K4_N8_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type cosine --sampling_type spatial --num_samples 8

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_11/vgg_1e-5_cos/CIFAR100/Loc_Col/ConvNN_Spat_K9_N8_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type cosine --sampling_type spatial --num_samples 8
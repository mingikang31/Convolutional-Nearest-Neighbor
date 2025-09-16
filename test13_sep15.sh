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
python vgg_main.py --layer Conv2d --kernel_size 3 --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5/CIFAR10-noise/Conv2d_K3_s42 --seed 42 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --noise 0.1


python vgg_main.py --layer Conv2d --kernel_size 3 --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5/CIFAR100-noise/Conv2d_K3_s42 --seed 42 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --dataset cifar100 --noise 0.1


## II. ConvNN K1-10 All Sampling - Loc, Col euclidean similarity

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_eucl/CIFAR10-noise/Loc_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc --aggregation_type Col --magnitude_type euclidean --noise 0.1

## III. ConvNN K1-10 All Sampling - Loc_Col, Col euclidean similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_eucl/CIFAR10-noise/LocCol_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --magnitude_type euclidean --noise 0.1


## IV. ConvNN K1-10 All Sampling - Loc_Col, Loc_Col euclidean similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_eucl/CIFAR10-noise/LocCol_LocCol/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type euclidean --noise 0.1

## V. ConvNN K1-10 All Sampling - Col, Col euclidean similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_eucl/CIFAR10-noise/Col_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Col --aggregation_type Col --dataset cifar10 --magnitude_type euclidean --noise 0.1

# CIFAR100
## I. Conv2d K3, K2

## II. ConvNN K1-10 All Sampling - Loc, Col euclidean similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_eucl/CIFAR100-noise/Loc_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100 --magnitude_type euclidean --noise 0.1

## III. ConvNN K1-10 All Sampling - Loc_Col, Col euclidean similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_eucl/CIFAR100-noise/LocCol_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100 --magnitude_type euclidean --noise 0.1

## IV. ConvNN K1-10 All Sampling - Loc_Col, Loc_Col euclidean similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_eucl/CIFAR100-noise/LocCol_LocCol/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type euclidean --noise 0.1

## V. ConvNN K1-10 All Sampling - Col, Col euclidean similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_eucl/CIFAR100-noise/Col_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Col --aggregation_type Col --dataset cifar100 --magnitude_type euclidean --noise 0.1



## II. ConvNN K1-10 All Sampling - Loc, Col cosine similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_cos/CIFAR10-noise/Loc_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc --aggregation_type Col --magnitude_type cosine --noise 0.1

## III. ConvNN K1-10 All Sampling - Loc_Col, Col cosine similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_cos/CIFAR10-noise/LocCol_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --noise 0.1

## IV. ConvNN K1-10 All Sampling - Loc_Col, Loc_Col cosine similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_cos/CIFAR10-noise/LocCol_LocCol/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --noise 0.1

# V. ConvNN K1-10 All Sampling - Col, Col cosine similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_cos/CIFAR10-noise/Col_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Col --aggregation_type Col --magnitude_type cosine --noise 0.1

# CIFAR100
## I. Conv2d K3, K2


## II. ConvNN K1-10 All Sampling - Loc, Col cosine similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_cos/CIFAR100-noise/Loc_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100 --magnitude_type cosine --noise 0.1

## III. ConvNN K1-10 All Sampling - Loc_Col, Col cosine similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_cos/CIFAR100-noise/LocCol_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100 --magnitude_type cosine --noise 0.1

## IV. ConvNN K1-10 All Sampling - Loc_Col, Loc_Col cosine similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_cos/CIFAR100-noise/LocCol_LocCol/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100 --magnitude_type cosine --noise 0.1


# V. ConvNN K1-10 All Sampling - Col, Col cosine similarity


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 100 --output_dir ./Output/Sep_15/vgg_1e-5_cos/CIFAR100-noise/Col_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 5 --lr_gamma 0.9 --lr 1e-4 --similarity_type Col --aggregation_type Col --dataset cifar100 --magnitude_type cosine --noise 0.1



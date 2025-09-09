#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=PT1_Exp_VGG
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
python vgg_main.py --layer Conv2d --kernel_size 3 --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Conv2d_K3_s42 --seed 42 --lr_step 2 --lr_gamma 0.95 --lr 1e-4

python vgg_main.py --layer Conv2d --kernel_size 2 --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Conv2d_K2_s42 --seed 42 --lr_step 2 --lr_gamma 0.95 --lr 1e-4

## II. ConvNN K1-10 All Sampling - Loc, Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Loc_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 2 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Loc_Col/ConvNN_All_K2_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 3 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Loc_Col/ConvNN_All_K3_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Loc_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 5 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Loc_Col/ConvNN_All_K5_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 6 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Loc_Col/ConvNN_All_K6_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 7 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Loc_Col/ConvNN_All_K7_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 8 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Loc_Col/ConvNN_All_K8_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Loc_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 10 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/Loc_Col/ConvNN_All_K10_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col 

## III. ConvNN K1-10 All Sampling - Loc_Col, Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 2 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_Col/ConvNN_All_K2_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 3 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_Col/ConvNN_All_K3_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 5 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_Col/ConvNN_All_K5_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 6 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_Col/ConvNN_All_K6_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 7 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_Col/ConvNN_All_K7_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 8 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_Col/ConvNN_All_K8_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col 

python vgg_main.py --layer ConvNN --K 10 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_Col/ConvNN_All_K10_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col 


## IV. ConvNN K1-10 All Sampling - Loc_Col, Loc_Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_LocCol/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col 

python vgg_main.py --layer ConvNN --K 2 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_LocCol/ConvNN_All_K2_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col 

python vgg_main.py --layer ConvNN --K 3 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_LocCol/ConvNN_All_K3_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col 

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_LocCol/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col 

python vgg_main.py --layer ConvNN --K 5 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_LocCol/ConvNN_All_K5_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col 

python vgg_main.py --layer ConvNN --K 6 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_LocCol/ConvNN_All_K6_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col 

python vgg_main.py --layer ConvNN --K 7 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_LocCol/ConvNN_All_K7_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col 

python vgg_main.py --layer ConvNN --K 8 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_LocCol/ConvNN_All_K8_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_LocCol/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col 

python vgg_main.py --layer ConvNN --K 10 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR10/LocCol_LocCol/ConvNN_All_K10_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col 

# CIFAR100 
## I. Conv2d K3, K2 

python vgg_main.py --layer Conv2d --kernel_size 3 --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Conv2d_K3_s42 --seed 42 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --dataset cifar100

python vgg_main.py --layer Conv2d --kernel_size 2 --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Conv2d_K2_s42 --seed 42 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --dataset cifar100

## II. ConvNN K1-10 All Sampling - Loc, Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Loc_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 2 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Loc_Col/ConvNN_All_K2_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 3 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Loc_Col/ConvNN_All_K3_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Loc_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 5 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Loc_Col/ConvNN_All_K5_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 6 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Loc_Col/ConvNN_All_K6_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 7 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Loc_Col/ConvNN_All_K7_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 8 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Loc_Col/ConvNN_All_K8_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Loc_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 10 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/Loc_Col/ConvNN_All_K10_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc --aggregation_type Col --dataset cifar100

## III. ConvNN K1-10 All Sampling - Loc_Col, Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_Col/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 2 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_Col/ConvNN_All_K2_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 3 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_Col/ConvNN_All_K3_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_Col/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 5 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_Col/ConvNN_All_K5_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 6 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_Col/ConvNN_All_K6_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 7 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_Col/ConvNN_All_K7_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 8 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_Col/ConvNN_All_K8_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_Col/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 10 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_Col/ConvNN_All_K10_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Col --dataset cifar100


## IV. ConvNN K1-10 All Sampling - Loc_Col, Loc_Col cosine similarity
python vgg_main.py --layer ConvNN --K 1 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_LocCol/ConvNN_All_K1_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 2 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_LocCol/ConvNN_All_K2_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 3 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_LocCol/ConvNN_All_K3_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 4 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_LocCol/ConvNN_All_K4_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 5 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_LocCol/ConvNN_All_K5_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 6 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_LocCol/ConvNN_All_K6_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 7 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_LocCol/ConvNN_All_K7_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 8 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_LocCol/ConvNN_All_K8_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_LocCol/ConvNN_All_K9_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100

python vgg_main.py --layer ConvNN --K 10 --sampling_type all --num_epochs 50 --output_dir ./Output/FINAL_TEST/vgg/CIFAR100/LocCol_LocCol/ConvNN_All_K10_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-4 --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cifar100


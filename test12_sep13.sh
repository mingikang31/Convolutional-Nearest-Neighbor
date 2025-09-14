#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=Test12_Lambda_Exp_VGG
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi

### VGG Experiments
# CIFAR10 
## III. ConvNN K1-10 All Sampling - Loc_Col, Col euclidean similarity

## Lambda = 0.5
python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_Col/ConvNN_All_K9_L_05_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type euclidean --lambda_param 0.5

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_05_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type euclidean --lambda_param 0.5



python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_Col/ConvNN_All_K9_L_05_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --lambda_param 0.5

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_05_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --lambda_param 0.5

## Lambda = 0.3 
python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_Col/ConvNN_All_K9_L_03_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type euclidean --lambda_param 0.3

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_03_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type euclidean --lambda_param 0.3


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_Col/ConvNN_All_K9_L_03_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --lambda_param 0.3

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_03_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --lambda_param 0.3


## Lambda = 0.7
python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_Col/ConvNN_All_K9_L_07_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type euclidean --lambda_param 0.7

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_07_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type euclidean --lambda_param 0.7


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_Col/ConvNN_All_K9_L_07_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --lambda_param 0.7

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_07_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --lambda_param 0.7

## lambda = 0.1 
python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_Col/ConvNN_All_K9_L_01_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type euclidean --lambda_param 0.1

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_01_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type euclidean --lambda_param 0.1


python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_Col/ConvNN_All_K9_L_01_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --lambda_param 0.1

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_01_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --lambda_param 0.1


## lambda = 0.9 
python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_Col/ConvNN_All_K9_L_09_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type euclidean --lambda_param 0.9

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_09_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type euclidean --lambda_param 0.9

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_Col/ConvNN_All_K9_L_09_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine --lambda_param 0.9

python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_09_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine --lambda_param 0.9


# ## Learned Parameter Lambda

# python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_Col/ConvNN_All_K9_L_param_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type euclidean

# python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_eucl/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_param_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type euclidean

# python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_Col/ConvNN_All_K9_L_param_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Col --magnitude_type cosine

# python vgg_main.py --layer ConvNN --K 9 --sampling_type all --num_epochs 30 --output_dir ./Output/Sep_13_lambda_pt2/vgg_1e-5_cos/CIFAR10/LocCol_LocCol/ConvNN_All_K9_L_param_s42 --seed 42 --padding 0 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --similarity_type Loc_Col --aggregation_type Loc_Col --magnitude_type cosine


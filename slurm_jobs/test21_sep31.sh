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

## ConvNN Attention 
python vgg_main.py --layer ConvNN_Attn --K 9 --padding 1 --sampling_type all --num_samples -1 --attention_dropout 0.1 --magnitude_type cosine --aggregation_type Col --output_dir ./Output/Sep_31_Attention/vgg_1e-5_cos/CIFAR10/ConvNN_Attn_All_K9_s42 --num_epochs 60 --seed 42 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 

python vgg_main.py --layer Branching_Attn --kernel_size 3 --K 9 --sampling_type all --num_epochs 60 --output_dir ./Output/Sep_31_Attention/vgg_1e-5_cos/CIFAR10/ConvAttnBranch_K9_r0250_s42 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-5 --aggregation_type Col --magnitude_type cosine --branch_ratio 0.250 --attention_dropout 0.1 --device mps --test_only

#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=CVPR1
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor


## VGG11 + VGG13 
## I. CIFAR10 
# 1. Baseline 
python main.py --model vgg11 --layer Conv2d --kernel_size 3 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG11/Conv2d_K3_s42

python main.py --model vgg13 --layer Conv2d --kernel_size 3 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG13/Conv2d_K3_s42

# 2. ConvNN 
python main.py --model vgg11 --layer ConvNN --K 9 --padding 1 --sampling_type all --num_samples -1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG11/ConvNN_All_K9_s42

python main.py --model vgg13 --layer ConvNN --K 9 --padding 1 --sampling_type all --num_samples -1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG13/ConvNN_All_K9_s42

python main.py --model vgg11 --layer ConvNN --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG11/ConvNN_Random_K9_N36_s42

python main.py --model vgg13 --layer ConvNN --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG13/ConvNN_Random_K9_N36_s42

python main.py --model vgg11 --layer ConvNN --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG11/ConvNN_Spatial_K9_N6_s42

python main.py --model vgg13 --layer ConvNN --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG13/ConvNN_Spatial_K9_N6_s42



# 3. Branching ConvNN
python main.py --model vgg11 --layer Branching --sampling_type all --num_samples -1 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG11/BranchingConvNN_All_K9_s42

python main.py --model vgg13 --layer Branching --sampling_type all --num_samples -1 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG13/BranchingConvNN_All_K9_s42

python main.py --model vgg11 --layer Branching --sampling_type random --num_samples 36 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG11/BranchingConvNN_Random_K9_N36_s42

python main.py --model vgg13 --layer Branching --sampling_type random --num_samples 36 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG13/BranchingConvNN_Random_K9_N36_s42

python main.py --model vgg11 --layer Branching --sampling_type spatial --num_samples 6 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG11/BranchingConvNN_Spatial_K9_N6_s42

python main.py --model vgg13 --layer Branching --sampling_type spatial --num_samples 6 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/VGG13/BranchingConvNN_Spatial_K9_N6_s42


## II. CIFAR100 
# 1. Baseline
python main.py --model vgg11 --layer Conv2d --kernel_size 3 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG11/Conv2d_K3_s42

python main.py --model vgg13 --layer Conv2d --kernel_size 3 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG13/Conv2d_K3_s42

# 2. ConvNN
python main.py --model vgg11 --layer ConvNN --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG11/ConvNN_K9_s42

python main.py --model vgg13 --layer ConvNN --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG13/ConvNN_K9_s42

python main.py --model vgg11 --layer ConvNN --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG11/ConvNN_Random_K9_N36_s42

python main.py --model vgg13 --layer ConvNN --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG13/ConvNN_Random_K9_N36_s42

python main.py --model vgg11 --layer ConvNN --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG11/ConvNN_Spatial_K9_N6_s42

python main.py --model vgg13 --layer ConvNN --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG13/ConvNN_Spatial_K9_N6_s42

# 3. Branching ConvNN
python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG11/BranchingConvNN_K9_s42

python main.py --model vgg13 --layer Branching --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG13/BranchingConvNN_K9_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG11/BranchingConvNN_Random_K9_N36_s42

python main.py --model vgg13 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG13/BranchingConvNN_Random_K9_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N6_s42

python main.py --model vgg13 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/VGG13/BranchingConvNN_Spatial_K9_N6_s42

### ResNet18 + ResNet34
# 1. Baseline 
python main.py --model resnet18 --layer Conv2d --kernel_size 3 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet18/Conv2d_K3_s42

python main.py --model resnet34 --layer Conv2d --kernel_size 3 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet34/Conv2d_K3_s42

# 2. ConvNN 
python main.py --model resnet18 --layer ConvNN --K 9 --padding 1 --sampling_type all --num_samples -1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet18/ConvNN_All_K9_s42

python main.py --model resnet34 --layer ConvNN --K 9 --padding 1 --sampling_type all --num_samples -1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet34/ConvNN_All_K9_s42

python main.py --model resnet18 --layer ConvNN --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet18/ConvNN_Random_K9_N36_s42

python main.py --model resnet34 --layer ConvNN --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet34/ConvNN_Random_K9_N36_s42

python main.py --model resnet18 --layer ConvNN --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet18/ConvNN_Spatial_K9_N6_s42

python main.py --model resnet34 --layer ConvNN --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet34/ConvNN_Spatial_K9_N6_s42

# 3. Branching ConvNN
python main.py --model resnet18 --layer Branching --sampling_type all --num_samples -1 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet18/BranchingConvNN_All_K9_s42

python main.py --model resnet34 --layer Branching --sampling_type all --num_samples -1 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet34/BranchingConvNN_All_K9_s42

python main.py --model resnet18 --layer Branching --sampling_type random --num_samples 36 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet18/BranchingConvNN_Random_K9_N36_s42

python main.py --model resnet34 --layer Branching --sampling_type random --num_samples 36 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet34/BranchingConvNN_Random_K9_N36_s42

python main.py --model resnet18 --layer Branching --sampling_type spatial --num_samples 6 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet18/BranchingConvNN_Spatial_K9_N6_s42

python main.py --model resnet34 --layer Branching --sampling_type spatial --num_samples 6 --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar10 --device cuda --seed 42 --output_dir ./Output/CIFAR10/ResNet34/BranchingConvNN_Spatial_K9_N6_s42


## II. CIFAR100 
# 1. Baseline
python main.py --model resnet18 --layer Conv2d --kernel_size 3 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet18/Conv2d_K3_s42

python main.py --model resnet34 --layer Conv2d --kernel_size 3 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet34/Conv2d_K3_s42

# 2. ConvNN
python main.py --model resnet18 --layer ConvNN --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet18/ConvNN_K9_s42

python main.py --model resnet34 --layer ConvNN --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet34/ConvNN_K9_s42

python main.py --model resnet18 --layer ConvNN --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet18/ConvNN_Random_K9_N36_s42

python main.py --model resnet34 --layer ConvNN --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet34/ConvNN_Random_K9_N36_s42

python main.py --model resnet18 --layer ConvNN --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet18/ConvNN_Spatial_K9_N6_s42

python main.py --model resnet34 --layer ConvNN --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet34/ConvNN_Spatial_K9_N6_s42

# 3. Branching ConvNN
python main.py --model resnet18 --layer Branching --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet18/BranchingConvNN_K9_s42

python main.py --model resnet34 --layer Branching --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet34/BranchingConvNN_K9_s42

python main.py --model resnet18 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet18/BranchingConvNN_Random_K9_N36_s42

python main.py --model resnet34 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet34/BranchingConvNN_Random_K9_N36_s42

python main.py --model resnet18 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet18/BranchingConvNN_Spatial_K9_N6_s42

python main.py --model resnet34 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/CIFAR100/ResNet34/BranchingConvNN_Spatial_K9_N6_s42



## III. Ablation Studies 
# K-Test on Branching VGG11 on CIFAR100 K 1 - 10
python main.py --model vgg11 --layer Branching --kernel_size 3 --K 1 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K1_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 2 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K2_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 3 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K3_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 4 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K4_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 5 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K5_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 6 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 7 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K7_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 8 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K8_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 10 --padding 1 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K10_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 1 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K1_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 2 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K2_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 3 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K3_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 4 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K4_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 5 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K5_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 6 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K6_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 7 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K7_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 8 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K8_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 10 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K10_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 1 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K1_N6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 2 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K2_N6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 3 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K3_N6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 4 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K4_N6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 5 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K5_N6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 6 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K6_N6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 7 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K7_N6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 8 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K8_N6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 10 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K10_N6_s42


# N-Test on Branching VGG11 on CIFAR100 N 

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 16 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N16_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 36 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N36_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 64 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N64_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 100 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N100_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 144 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N144_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 196 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N196_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 256 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N256_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 324 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N324_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 400 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N400_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 484 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N484_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 576 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N576_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 676 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N676_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 784 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N784_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type random --num_samples 900 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Random_K9_N900_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 4 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N4_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 6 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N6_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 8 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N8_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 10 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N10_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 12 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N12_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 14 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N14_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 16 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N16_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 18 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N18_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 20 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N20_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 22 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N22_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 24 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N24_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 26 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N26_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 28 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N28_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --sampling_type spatial --num_samples 30 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_Spatial_K9_N30_s42

# Ablation studies branching ratio 
python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.000 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br0000_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.125 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br0125_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.250 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br0250_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.375 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br0375_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.500 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br0500_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.625 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br0625_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.750 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br0750_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 0.875 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br0875_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Col --aggregation_type Col --branch_ratio 1.000 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_col_col_br1000_s42



python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Col --branch_ratio 0.000 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_col_br0000_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Col --branch_ratio 0.125 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_col_br0125_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Col --branch_ratio 0.250 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_col_br0250_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Col --branch_ratio 0.375 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_col_br0375_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Col --branch_ratio 0.500 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_col_br0500_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Col --branch_ratio 0.625 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_col_br0625_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Col --branch_ratio 0.750 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_col_br0750_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Col --branch_ratio 0.875 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_col_br0875_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Col --branch_ratio 1.000 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_col_br1000_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Loc_Col --branch_ratio 0.000 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_loccol_br0000_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Loc_Col --branch_ratio 0.125 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_loccol_br0125_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Loc_Col --branch_ratio 0.250 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_loccol_br0250_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Loc_Col --branch_ratio 0.375 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_loccol_br0375_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Loc_Col --branch_ratio 0.500 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_loccol_br0500_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Loc_Col --branch_ratio 0.625 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_loccol_br0625_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Loc_Col --branch_ratio 0.750 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_loccol_br0750_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Loc_Col --branch_ratio 0.875 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_loccol_br0875_s42

python main.py --model vgg11 --layer Branching --kernel_size 3 --K 9 --padding 1 --similarity_type Loc_Col --aggregation_type Loc_Col --branch_ratio 1.000 --criterion CrossEntropy --batch_size 128 --num_epochs 200 --optimizer adamw --weight_decay 1e-2 --lr 1e-3 --scheduler cosine --dataset cifar100 --device cuda --seed 42 --output_dir ./Output/Ablation-CIFAR100/VGG11/BranchingConvNN_K9_loccol_loccol_br1000_s42














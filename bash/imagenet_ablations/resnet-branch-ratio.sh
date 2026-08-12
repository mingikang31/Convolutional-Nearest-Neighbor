#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=480G
#SBATCH -p mixed --gres=gpu:pro6000:1
#SBATCH --cpus-per-gpu=80
#SBATCH --job-name=imgnet-br-ratio
#SBATCH --time=720:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
set -u
conda activate torch-pro6000

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor 

BRANCH_RATIOS=("0.125" "0.250" "0.375" "0.500" "0.625" "0.750" "0.875" "0.000" "1.000")

for br in "${BRANCH_RATIOS[@]}"; do
    br_fmt=$(awk -v br="$br" 'BEGIN{printf "%04d", br*1000}')

    output_dir="./ImageNet-Output/ResNet-50/Branching_All_K9_Ks3_br${br_fmt}"

    echo "=== Starting run: branch_ratio=${br} -> ${output_dir} ==="

    if ! python main.py \
        --model resnet50 \
        --layer Branching \
        --K 9 \
        --kernel_size 3 \
        --padding 1 \
        --sampling_type all \
        --branch_ratio "$br" \
        --num_samples -1 \
        --dataset imagenet1k \
        --compile \
        --use_amp \
        --num_workers 12 \
        --pin_memory \
        --batch_size 512 \
        --data_path /mnt/research/j.farias/mkang2/Datasets \
        --output_dir "$output_dir"; then 
        echo "!!! Run failed for branch_ratio=${br}, continuing to next ratio !!!" >&2
    fi
done
 
echo "=== Sweep complete ==="

#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=CVPR-K
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor
source activate mingi

### K-Test for CVPR paper 

# Configuration
DATASETS=("cifar10" "cifar100")
BRANCH_RATIOS=("0.000" "0.500" "1.000")
KERNEL_SIZES=("1" "2" "3")                          
K_VALUES=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")  
LR="1e-4"                                         

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#BRANCH_RATIOS[@]} * ${#KERNEL_SIZES[@]} * ${#K_VALUES[@]}))
COUNT=0
FAILED=0

echo "=========================================="
echo "K-Test Configuration"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Datasets: ${DATASETS[@]}"
echo "Branch ratios: ${BRANCH_RATIOS[@]}"
echo "Kernel sizes: ${KERNEL_SIZES[@]}"
echo "K values: ${K_VALUES[@]}"
echo "Learning rate: $LR"
echo "=========================================="
echo ""

# Main loop
for dataset in "${DATASETS[@]}"; do
    for br in "${BRANCH_RATIOS[@]}"; do
        for ks in "${KERNEL_SIZES[@]}"; do
            for k in "${K_VALUES[@]}"; do
                
                COUNT=$((COUNT + 1))
                
                # Format branch ratio (0.5 → 0500)
                br_int=$(echo "$br * 10000 / 10" | bc)
                br_fmt=$(printf "%04d" $br_int)
                
                # Determine padding based on K value
                if [ "$k" -lt 5 ]; then
                    padding=0
                else
                    padding=1
                fi
                
                # Create output directory
                output_dir="./Final_Output/K_test/VGG11-$(echo $dataset | awk '{print toupper($0)}')/BranchingConvNN_K${k}_KS${ks}_col_col_br${br_fmt}_s42"
                
                echo "[$COUNT/$TOTAL] Dataset=$dataset | K=$k | KS=$ks | BR=$br | Padding=$padding"
                echo "Output: $output_dir"
                
                # Single python call with padding set conditionally
                python main.py \
                    --model vgg11 \
                    --layer Branching \
                    --kernel_size $ks \
                    --K $k \
                    --padding $padding \
                    --similarity_type Col \
                    --aggregation_type Col \
                    --branch_ratio $br \
                    --criterion CrossEntropy \
                    --optimizer adamw \
                    --weight_decay 0.01 \
                    --lr $LR \
                    --scheduler none \
                    --clip_grad_norm 1.0 \
                    --dataset $dataset \
                    --device cuda \
                    --seed 42 \
                    --output_dir $output_dir
                
                # Check if experiment succeeded
                if [ $? -eq 0 ]; then
                    echo "✓ Experiment $COUNT succeeded"
                else
                    echo "✗ Experiment $COUNT failed"
                    FAILED=$((FAILED + 1))
                fi
                echo ""
                
            done
        done
    done
done

echo "=========================================="
echo "K-Test Complete!"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="
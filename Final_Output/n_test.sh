#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=CVPR-N-Test
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor
source activate mingi

### N-Test for CVPR paper 

# Configuration
DATASETS=("cifar10" "cifar100")
BRANCH_RATIOS=("0.500" "1.000")
NS=("2" "4" "6" "8" "10" "12" "14" "16" "18" "20" "22" "24" "26" "28" "30" "32")
SAMPLING_TYPES=("random" "spatial")  # Fixed: renamed from SAMPLING_TYPE
LR="1e-4"

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#BRANCH_RATIOS[@]} * ${#NS[@]} * ${#SAMPLING_TYPES[@]}))
COUNT=0
FAILED=0

echo "=========================================="
echo "N-Test Configuration"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Datasets: ${DATASETS[@]}"
echo "Branch ratios: ${BRANCH_RATIOS[@]}"
echo "N values: ${NS[@]}"
echo "Sampling types: ${SAMPLING_TYPES[@]}"
echo "Learning rate: $LR"
echo "=========================================="
echo ""

# Main loop
for dataset in "${DATASETS[@]}"; do
    for br in "${BRANCH_RATIOS[@]}"; do
        for ns in "${NS[@]}"; do
            for st in "${SAMPLING_TYPES[@]}"; do
                
                COUNT=$((COUNT + 1))
                
                # Format branch ratio (0.5 → 0500)
                br_int=$(echo "$br * 10000 / 10" | bc)
                br_fmt=$(printf "%04d" $br_int)
                
                # Calculate num_samples based on sampling type
                if [ "$st" = "random" ]; then
                    num_samples=$((ns * ns))  # N squared for random
                else
                    num_samples=$ns  # N for spatial
                fi
                
                # Create output directory
                output_dir="./Final_Output/N_test/VGG11-$(echo $dataset | awk '{print toupper($0)}')/BranchingConvNN_${st}_NS${ns}_col_col_br${br_fmt}_s42"

                echo "[$COUNT/$TOTAL] Dataset=$dataset | ST=$st | NS=$ns | Num_Samples=$num_samples | BR=$br"
                echo "Output: $output_dir"
                
                # Single python call
                python main.py \
                    --model vgg11 \
                    --layer Branching \
                    --kernel_size 3 \
                    --K 9 \
                    --padding 1 \
                    --sampling_type $st \
                    --num_samples $num_samples \
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
echo "N-Test Complete!"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="
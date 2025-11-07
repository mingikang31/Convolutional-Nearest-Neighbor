#!/bin/bash 

### Conv-Test for CVPR paper 

cd /home/exouser/Convolutional-Nearest-Neighbor/
### Branch Ratio Test for CVPR paper

# Configuration
DATASETS=("cifar10" "cifar100")
BRANCH_RATIOS=("0.000" "0.125" "0.250" "0.375" "0.500" "0.625" "0.750" "0.875" "1.000")
SIMILARITY_TYPES=("Col" "Loc_Col" "Loc_Col")
AGGREGATION_TYPES=("Col" "Col" "Loc_Col")
LR="1e-4"

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#BRANCH_RATIOS[@]} * ${#SIMILARITY_TYPES[@]}))
COUNT=0
FAILED=0

echo "=========================================="
echo "Branch Ratio Test Configuration"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Datasets: ${DATASETS[@]}"
echo "Branch ratios: ${BRANCH_RATIOS[@]}"
echo "Similarity types: ${SIMILARITY_TYPES[@]}"
echo "Aggregation types: ${AGGREGATION_TYPES[@]}"
echo "Learning rate: $LR"
echo "=========================================="
echo ""

# Main loop
for dataset in "${DATASETS[@]}"; do
    for br in "${BRANCH_RATIOS[@]}"; do
        for idx in "${!SIMILARITY_TYPES[@]}"; do
            
            COUNT=$((COUNT + 1))
            
            # Get current similarity and aggregation types
            sim_type="${SIMILARITY_TYPES[$idx]}"
            agg_type="${AGGREGATION_TYPES[$idx]}"
            
            # Format branch ratio (0.5 → 0500)
            br_int=$(echo "$br * 10000 / 10" | bc)
            br_fmt=$(printf "%04d" $br_int)
            
            # Create output directory
            output_dir="./Final_Output/BR_test_eucl/VGG11-$(echo $dataset | awk '{print toupper($0)}')/BranchingConvNN_All_K9_${sim_type}_${agg_type}_br${br_fmt}_s42"
            
            echo "[$COUNT/$TOTAL] Dataset=$dataset | BR=$br | Sim=$sim_type | Agg=$agg_type"
            echo "Output: $output_dir"
            
            # Single python call
            python main.py \
                --model vgg11 \
                --layer Branching \
                --kernel_size 3 \
                --K 9 \
                --padding 1 \
                --sampling_type all \
                --num_samples -1 \
                --magnitude_type euclidean \
                --similarity_type $sim_type \
                --aggregation_type $agg_type \
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

echo "=========================================="
echo "Branch Ratio Test Complete!"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="
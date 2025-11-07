#!/bin/bash 

### Conv-Test for CVPR paper 

cd /home/exouser/Convolutional-Nearest-Neighbor/

# Configuration
DATASETS=("cifar10" "cifar100")
LEARNING_RATES=("1e-3" "5e-4" "1e-4" "5e-5" "1e-5")
BRANCH_RATIOS=("0.500" "1.000")

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#LEARNING_RATES[@]} * ${#BRANCH_RATIOS[@]}))
COUNT=0

# Main loop
for dataset in "${DATASETS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for br in "${BRANCH_RATIOS[@]}"; do
            
            COUNT=$((COUNT + 1))
            
            # Format branch ratio (0.5 → 0500)
            br_int=$(echo "$br * 10000 / 10" | bc)
            br_fmt=$(printf "%04d" $br_int)
            
            # Create output directory
            output_dir="./Final_Output/loss_test_cos/VGG11-$(echo $dataset | awk '{print toupper($0)}')/lr_${lr}/BranchingConvNNAttn_K9_col_col_br${br_fmt}_s42"
            
            echo ""
            echo "========== Experiment $COUNT/$TOTAL =========="
            echo "Dataset: $dataset | LR: $lr | Branch Ratio: $br"
            echo "Output: $output_dir"
            echo "=========================================="
            
            python main.py \
                --model vgg11 \
                --layer Branching_Attn \
                --kernel_size 3 \
                --K 9 \
                --padding 1 \
                --magnitude_type cosine \
                --similarity_type Col \
                --aggregation_type Col \
                --branch_ratio $br \
                --criterion CrossEntropy \
                --optimizer adamw \
                --weight_decay 0.01 \
                --lr $lr \
                --scheduler none \
                --clip_grad_norm 1.0 \
                --dataset $dataset \
                --device cuda \
                --seed 42 \
                --output_dir $output_dir
            
            # Check if experiment succeeded
            if [ $? -eq 0 ]; then
                echo "✓ Completed successfully"
            else
                echo "✗ Failed - continuing to next experiment"
            fi
        done
    done
done

echo ""
echo "All $TOTAL experiments completed!"
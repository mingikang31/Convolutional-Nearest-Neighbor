#!/bin/bash 
cd /home/exouser/Convolutional-Nearest-Neighbor/


# Configuration
DATASETS=("cifar10" "cifar100")
BRANCH_RATIOS=("0.500")
KERNEL_SIZES=("1" "2" "3")                          
K_VALUES=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")  
LR="1e-4"                                         

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#BRANCH_RATIOS[@]} * ${#KERNEL_SIZES[@]} * ${#K_VALUES[@]} + ${#DATASETS[@]} * 3 + ${#DATASETS[@]} * ${#K_VALUES[@]}))
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

# Main loop - Branch ratio 0.5 with all K and KS combinations
for dataset in "${DATASETS[@]}"; do
    for br in "${BRANCH_RATIOS[@]}"; do
        for ks in "${KERNEL_SIZES[@]}"; do
            for k in "${K_VALUES[@]}"; do
                
                COUNT=$((COUNT + 1))
                
                # Format branch ratio (0.5 → 0500)
                br_int=$(echo "$br * 10000 / 10" | bc)
                br_fmt=$(printf "%04d" $br_int)
                
                # Create output directory
                output_dir="./Final_Output/K_test1_eucl/VGG11-$(echo $dataset | awk '{print toupper($0)}')/BranchingConvNN_K${k}_KS${ks}_col_col_br${br_fmt}_s42"
                
                echo "[$COUNT/$TOTAL] Dataset=$dataset | K=$k | KS=$ks | BR=$br"
                echo "Output: $output_dir"
                
                # Single python call with padding set conditionally
                python main.py \
                    --model vgg11 \
                    --layer Branching \
                    --kernel_size $ks \
                    --K $k \
                    --padding 1 \
                    --magnitude_type euclidean \
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
                    --output_dir $output_dir \
                    --num_epochs 100
                
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
echo "Pure Convolution Baselines (br=0.000)"
echo "=========================================="

# CIFAR 10 
COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar10 | K=0 | KS=3 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 3 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar10 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test1_eucl/VGG11-CIFAR10/BranchingConvNN_K0_KS3_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar10 | K=0 | KS=2 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 2 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar10 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test1_eucl/VGG11-CIFAR10/BranchingConvNN_K0_KS2_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar10 | K=0 | KS=1 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 1 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar10 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test1_eucl/VGG11-CIFAR10/BranchingConvNN_K0_KS1_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""


# CIFAR 100

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar100 | K=0 | KS=3 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 3 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar100 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test1_eucl/VGG11-CIFAR100/BranchingConvNN_K0_KS3_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar100 | K=0 | KS=2 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 2 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar100 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test1_eucl/VGG11-CIFAR100/BranchingConvNN_K0_KS2_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar100 | K=0 | KS=1 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 1 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar100 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test1_eucl/VGG11-CIFAR100/BranchingConvNN_K0_KS1_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""


# Configuration
DATASETS=("cifar10" "cifar100")
BRANCH_RATIOS=("0.500")
KERNEL_SIZES=("1" "2" "3")                          
K_VALUES=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")  
LR="1e-4"                                         

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#BRANCH_RATIOS[@]} * ${#KERNEL_SIZES[@]} * ${#K_VALUES[@]} + ${#DATASETS[@]} * 3 + ${#DATASETS[@]} * ${#K_VALUES[@]}))
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

# Main loop - Branch ratio 0.5 with all K and KS combinations
for dataset in "${DATASETS[@]}"; do
    for br in "${BRANCH_RATIOS[@]}"; do
        for ks in "${KERNEL_SIZES[@]}"; do
            for k in "${K_VALUES[@]}"; do
                
                COUNT=$((COUNT + 1))
                
                # Format branch ratio (0.5 → 0500)
                br_int=$(echo "$br * 10000 / 10" | bc)
                br_fmt=$(printf "%04d" $br_int)
                
                # Create output directory
                output_dir="./Final_Output/K_test2_eucl/VGG11-$(echo $dataset | awk '{print toupper($0)}')/BranchingConvNN_K${k}_KS${ks}_col_col_br${br_fmt}_s0"
                
                echo "[$COUNT/$TOTAL] Dataset=$dataset | K=$k | KS=$ks | BR=$br"
                echo "Output: $output_dir"
                
                # Single python call with padding set conditionally
                python main.py \
                    --model vgg11 \
                    --layer Branching \
                    --kernel_size $ks \
                    --K $k \
                    --padding 1 \
                    --magnitude_type euclidean \
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
                    --seed 0 \
                    --output_dir $output_dir \
                    --num_epochs 100
                
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
echo "Pure Convolution Baselines (br=0.000)"
echo "=========================================="

# CIFAR 10 
COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar10 | K=0 | KS=3 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 3 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar10 \
    --device cuda \
    --seed 0 \
    --output_dir ./Final_Output/K_test2_eucl/VGG11-CIFAR10/BranchingConvNN_K0_KS3_col_col_br0000_s0 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar10 | K=0 | KS=2 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 2 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar10 \
    --device cuda \
    --seed 0 \
    --output_dir ./Final_Output/K_test2_eucl/VGG11-CIFAR10/BranchingConvNN_K0_KS2_col_col_br0000_s0 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar10 | K=0 | KS=1 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 1 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar10 \
    --device cuda \
    --seed 0 \
    --output_dir ./Final_Output/K_test2_eucl/VGG11-CIFAR10/BranchingConvNN_K0_KS1_col_col_br0000_s0 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""


# CIFAR 100

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar100 | K=0 | KS=3 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 3 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar100 \
    --device cuda \
    --seed 0 \
    --output_dir ./Final_Output/K_test2_eucl/VGG11-CIFAR100/BranchingConvNN_K0_KS3_col_col_br0000_s0 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar100 | K=0 | KS=2 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 2 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar100 \
    --device cuda \
    --seed 0 \
    --output_dir ./Final_Output/K_test2_eucl/VGG11-CIFAR100/BranchingConvNN_K0_KS2_col_col_br0000_s0 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar100 | K=0 | KS=1 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 1 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar100 \
    --device cuda \
    --seed 0 \
    --output_dir ./Final_Output/K_test2_eucl/VGG11-CIFAR100/BranchingConvNN_K0_KS1_col_col_br0000_s0 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""



DATASETS=("cifar10" "cifar100")
BRANCH_RATIOS=("0.500")
KERNEL_SIZES=("1" "2" "3")                          
K_VALUES=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")  
LR="1e-4"                                         

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#BRANCH_RATIOS[@]} * ${#KERNEL_SIZES[@]} * ${#K_VALUES[@]} + ${#DATASETS[@]} * 3 + ${#DATASETS[@]} * ${#K_VALUES[@]}))
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

# Main loop - Branch ratio 0.5 with all K and KS combinations
for dataset in "${DATASETS[@]}"; do
    for br in "${BRANCH_RATIOS[@]}"; do
        for ks in "${KERNEL_SIZES[@]}"; do
            for k in "${K_VALUES[@]}"; do
                
                COUNT=$((COUNT + 1))
                
                # Format branch ratio (0.5 → 0500)
                br_int=$(echo "$br * 10000 / 10" | bc)
                br_fmt=$(printf "%04d" $br_int)
                
                # Create output directory
                output_dir="./Final_Output/K_test3_eucl/VGG11-$(echo $dataset | awk '{print toupper($0)}')/BranchingConvNN_K${k}_KS${ks}_col_col_br${br_fmt}_s42"
                
                echo "[$COUNT/$TOTAL] Dataset=$dataset | K=$k | KS=$ks | BR=$br"
                echo "Output: $output_dir"
                
                # Single python call with padding set conditionally
                python main.py \
                    --model vgg11 \
                    --layer Branching \
                    --kernel_size $ks \
                    --K $k \
                    --padding 1 \
                    --magnitude_type euclidean \
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
                    --output_dir $output_dir \
                    --num_epochs 100
                
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
echo "Pure Convolution Baselines (br=0.000)"
echo "=========================================="

# CIFAR 10 
COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar10 | K=0 | KS=3 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 3 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar10 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test3_eucl/VGG11-CIFAR10/BranchingConvNN_K0_KS3_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar10 | K=0 | KS=2 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 2 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar10 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test3_eucl/VGG11-CIFAR10/BranchingConvNN_K0_KS2_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar10 | K=0 | KS=1 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 1 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar10 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test3_eucl/VGG11-CIFAR10/BranchingConvNN_K0_KS1_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""


# CIFAR 100

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar100 | K=0 | KS=3 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 3 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar100 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test3_eucl/VGG11-CIFAR100/BranchingConvNN_K0_KS3_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar100 | K=0 | KS=2 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 2 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar100 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test3_eucl/VGG11-CIFAR100/BranchingConvNN_K0_KS2_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""

COUNT=$((COUNT + 1))
echo "[$COUNT/$TOTAL] Dataset=cifar100 | K=0 | KS=1 | BR=0.000 (Pure Conv)"
python main.py \
    --model vgg11 \
    --layer Branching \
    --kernel_size 1 \
    --K 1 \
    --branch_ratio 0.000 \
    --criterion CrossEntropy \
    --optimizer adamw \
    --weight_decay 0.01 \
    --lr $LR \
    --scheduler none \
    --clip_grad_norm 1.0 \
    --dataset cifar100 \
    --device cuda \
    --seed 42 \
    --output_dir ./Final_Output/K_test3_eucl/VGG11-CIFAR100/BranchingConvNN_K0_KS1_col_col_br0000_s42 \
    --num_epochs 100

if [ $? -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi
echo ""
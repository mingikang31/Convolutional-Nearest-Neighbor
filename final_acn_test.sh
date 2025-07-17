#!/bin/bash

### Table 1 - Baseline Models ###
# # - Configuration: 3 layer, 16, 32, 64 channels
# Baseline models: 
# 1. Regular Conv2d 16, 32, 64 
python allconvnet_main.py --layer Conv2d --kernel_size 3 --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/Baseline/Conv2d

# 2. Regular Conv2d more chans: 64, 128, 256
python allconvnet_main.py --layer Conv2d --kernel_size 3 --num_layers 3 --channels 64 128 256 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/Baseline/Conv2d_more_chans


### Table 2 - Vary K Tests ### 
# # ConvNN All Samples 
# 1. ConvNN All Samples K = 1
python allconvnet_main.py --layer ConvNN --K 1 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_K1

# 2. ConvNN All Samples K = 2 
python allconvnet_main.py --layer ConvNN --K 2 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_K2

# 3. ConvNN All Samples K = 3
python allconvnet_main.py --layer ConvNN --K 3 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_K3    

# 4. ConvNN All Samples K = 4
python allconvnet_main.py --layer ConvNN --K 4 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_K4

# 5. ConvNN All Samples K = 5
python allconvnet_main.py --layer ConvNN --K 5 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_K5   

# 6. ConvNN All Samples K = 6
python allconvnet_main.py --layer ConvNN --K 6 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_K6    

# 7. ConvNN All Samples K = 7
python allconvnet_main.py --layer ConvNN --K 7 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_K7    

# 8. ConvNN All Samples K = 8   
python allconvnet_main.py --layer ConvNN --K 8 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_K8

# 9. ConvNN All Samples K = 9
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_K9

# 10. ConvNN All Samples K = 10
python allconvnet_main.py --layer ConvNN --K 10 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_K10

# # ConvNN Random Samples
# 1. ConvNN Random Samples K = 1
python allconvnet_main.py --layer ConvNN --K 1 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_K1

# 2. ConvNN Random Samples K = 2
python allconvnet_main.py --layer ConvNN --K 2 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_K2

# 3. ConvNN Random Samples K = 3
python allconvnet_main.py --layer ConvNN --K 3 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_K3


# 4. ConvNN Random Samples K = 4
python allconvnet_main.py --layer ConvNN --K 4 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_K4

# 5. ConvNN Random Samples K = 5
python allconvnet_main.py --layer ConvNN --K 5 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_K5

# 6. ConvNN Random Samples K = 6
python allconvnet_main.py --layer ConvNN --K 6 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_K6

# 7. ConvNN Random Samples K = 7
python allconvnet_main.py --layer ConvNN --K 7 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_K7

# 8. ConvNN Random Samples K = 8
python allconvnet_main.py --layer ConvNN --K 8 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_K8

# 9. ConvNN Random Samples K = 9
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_K9

# 10. ConvNN Random Samples K = 10
python allconvnet_main.py --layer ConvNN --K 10 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_K10

# # ConvNN Spatial Samples
# 1. ConvNN Spatial Samples K = 1
python allconvnet_main.py --layer ConvNN --K 1 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_K1

# 2. ConvNN Spatial Samples K = 2
python allconvnet_main.py --layer ConvNN --K 2 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_K2

# 3. ConvNN Spatial Samples K = 3
python allconvnet_main.py --layer ConvNN --K 3 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_K3

# 4. ConvNN Spatial Samples K = 4
python allconvnet_main.py --layer ConvNN --K 4 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_K4

# 5. ConvNN Spatial Samples K = 5
python allconvnet_main.py --layer ConvNN --K 5 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_K5

# 6. ConvNN Spatial Samples K = 6
python allconvnet_main.py --layer ConvNN --K 6 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_K6

# 7. ConvNN Spatial Samples K = 7
python allconvnet_main.py --layer ConvNN --K 7 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_K7

# 8. ConvNN Spatial Samples K = 8
python allconvnet_main.py --layer ConvNN --K 8 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_K8    

# 9. ConvNN Spatial Samples K = 9
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_K9

# 10. ConvNN Spatial Samples K = 10
python allconvnet_main.py --layer ConvNN --K 10 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_K10  

# # ConvNN All Samples Coordinate Encoding 
# 1. ConvNN All Samples Coordinate Encoding K = 1
python allconvnet_main.py --layer ConvNN --K 1 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_Coord_K1 --coordinate_encoding

# 2. ConvNN All Samples Coordinate Encoding K = 2
python allconvnet_main.py --layer ConvNN --K 2 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_Coord_K2 --coordinate_encoding

# 3. ConvNN All Samples Coordinate Encoding K = 3
python allconvnet_main.py --layer ConvNN --K 3 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_Coord_K3 --coordinate_encoding 

# 4. ConvNN All Samples Coordinate Encoding K = 4
python allconvnet_main.py --layer ConvNN --K 4 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_Coord_K4 --coordinate_encoding

# 5. ConvNN All Samples Coordinate Encoding K = 5
python allconvnet_main.py --layer ConvNN --K 5 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_Coord_K5 --coordinate_encoding

# 6. ConvNN All Samples Coordinate Encoding K = 6
python allconvnet_main.py --layer ConvNN --K 6 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_Coord_K6 --coordinate_encoding

# 7. ConvNN All Samples Coordinate Encoding K = 7
python allconvnet_main.py --layer ConvNN --K 7 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_Coord_K7 --coordinate_encoding

# 8. ConvNN All Samples Coordinate Encoding K = 8
python allconvnet_main.py --layer ConvNN --K 8 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_Coord_K8 --coordinate_encoding

# 9. ConvNN All Samples Coordinate Encoding K = 9
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_Coord_K9 --coordinate_encoding

# 10. ConvNN All Samples Coordinate Encoding K = 10
python allconvnet_main.py --layer ConvNN --K 10 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_All_Coord_K10 --coordinate_encoding

# # ConvNN Random Samples Coordinate Encoding
# 1. ConvNN Random Samples Coordinate Encoding K = 1
python allconvnet_main.py --layer ConvNN --K 1 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_Coord_K1 --coordinate_encoding

# 2. ConvNN Random Samples Coordinate Encoding K = 2
python allconvnet_main.py --layer ConvNN --K 2 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_Coord_K2 --coordinate_encoding

# 3. ConvNN Random Samples Coordinate Encoding K = 3
python allconvnet_main.py --layer ConvNN --K 3 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_Coord_K3 --coordinate_encoding  

# 4. ConvNN Random Samples Coordinate Encoding K = 4
python allconvnet_main.py --layer ConvNN --K 4 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_Coord_K4 --coordinate_encoding  

# 5. ConvNN Random Samples Coordinate Encoding K = 5
python allconvnet_main.py --layer ConvNN --K 5 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_Coord_K5 --coordinate_encoding

# 6. ConvNN Random Samples Coordinate Encoding K = 6
python allconvnet_main.py --layer ConvNN --K 6 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_Coord_K6 --coordinate_encoding

# 7. ConvNN Random Samples Coordinate Encoding K = 7
python allconvnet_main.py --layer ConvNN --K 7 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_Coord_K7 --coordinate_encoding

# 8. ConvNN Random Samples Coordinate Encoding K = 8
python allconvnet_main.py --layer ConvNN --K 8 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_Coord_K8 --coordinate_encoding

# 9. ConvNN Random Samples Coordinate Encoding K = 9
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_Coord_K9 --coordinate_encoding

# 10. ConvNN Random Samples Coordinate Encoding K = 10
python allconvnet_main.py --layer ConvNN --K 10 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Random_Coord_K10 --coordinate_encoding    

# # ConvNN Spatial Samples Coordinate Encoding
# 1. ConvNN Spatial Samples Coordinate Encoding K = 1
python allconvnet_main.py --layer ConvNN --K 1 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_Coord_K1 --coordinate_encoding

# 2. ConvNN Spatial Samples Coordinate Encoding K = 2
python allconvnet_main.py --layer ConvNN --K 2 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_Coord_K2 --coordinate_encoding

# 3. ConvNN Spatial Samples Coordinate Encoding K = 3
python allconvnet_main.py --layer ConvNN --K 3 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_Coord_K3 --coordinate_encoding

# 4. ConvNN Spatial Samples Coordinate Encoding K = 4
python allconvnet_main.py --layer ConvNN --K 4 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_Coord_K4 --coordinate_encoding

# 5. ConvNN Spatial Samples Coordinate Encoding K = 5
python allconvnet_main.py --layer ConvNN --K 5 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_Coord_K5 --coordinate_encoding

# 6. ConvNN Spatial Samples Coordinate Encoding K = 6
python allconvnet_main.py --layer ConvNN --K 6 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_Coord_K6 --coordinate_encoding

# 7. ConvNN Spatial Samples Coordinate Encoding K = 7
python allconvnet_main.py --layer ConvNN --K 7 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_Coord_K7 --coordinate_encoding

# 8. ConvNN Spatial Samples Coordinate Encoding K = 8
python allconvnet_main.py --layer ConvNN --K 8 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_Coord_K8 --coordinate_encoding

# 9. ConvNN Spatial Samples Coordinate Encoding K = 9
python allconvnet_main.py --layer ConvNN --K 9 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_Coord_K9 --coordinate_encoding

# 10. ConvNN Spatial Samples Coordinate Encoding K = 10
python allconvnet_main.py --layer ConvNN --K 10 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Spatial_Coord_K10 --coordinate_encoding

# # ConvNN_Attn All Samples K = 1
python allconvnet_main.py --layer ConvNN_Attn --K 1 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_K1

# # ConvNN_Attn All Samples K = 2
python allconvnet_main.py --layer ConvNN_Attn --K 2 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_K2

# # ConvNN_Attn All Samples K = 3
python allconvnet_main.py --layer ConvNN_Attn --K 3 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_K3

# # ConvNN_Attn All Samples K = 4
python allconvnet_main.py --layer ConvNN_Attn --K 4 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_K4

# # ConvNN_Attn All Samples K = 5
python allconvnet_main.py --layer ConvNN_Attn --K 5 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_K5

# # ConvNN_Attn All Samples K = 6
python allconvnet_main.py --layer ConvNN_Attn --K 6 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_K6

# # ConvNN_Attn All Samples K = 7
python allconvnet_main.py --layer ConvNN_Attn --K 7 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_K7

# # ConvNN_Attn All Samples K = 8
python allconvnet_main.py --layer ConvNN_Attn --K 8 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_K8

# # ConvNN_Attn All Samples K = 9
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_K9   

# # ConvNN_Attn All Samples K = 10
python allconvnet_main.py --layer ConvNN_Attn --K 10 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_K10

# # ConvNN_Attn Random Samples K = 1
python allconvnet_main.py --layer ConvNN_Attn --K 1 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_K1

# # ConvNN_Attn Random Samples K = 2
python allconvnet_main.py --layer ConvNN_Attn --K 2 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_K2

# # ConvNN_Attn Random Samples K = 3
python allconvnet_main.py --layer ConvNN_Attn --K 3 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_K3

# # ConvNN_Attn Random Samples K = 4
python allconvnet_main.py --layer ConvNN_Attn --K 4 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_K4

# # ConvNN_Attn Random Samples K = 5
python allconvnet_main.py --layer ConvNN_Attn --K 5 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_K5   

# # ConvNN_Attn Random Samples K = 6
python allconvnet_main.py --layer ConvNN_Attn --K 6 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_K6

# # ConvNN_Attn Random Samples K = 7
python allconvnet_main.py --layer ConvNN_Attn --K 7 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_K7

# # ConvNN_Attn Random Samples K = 8
python allconvnet_main.py --layer ConvNN_Attn --K 8 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_K8

# # ConvNN_Attn Random Samples K = 9
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_K9

# # ConvNN_Attn Random Samples K = 10
python allconvnet_main.py --layer ConvNN_Attn --K 10 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_K10

# # ConvNN_Attn Spatial Samples K = 1
python allconvnet_main.py --layer ConvNN_Attn --K 1 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_K1

# # ConvNN_Attn Spatial Samples K = 2
python allconvnet_main.py --layer ConvNN_Attn --K 2 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_K2

# # ConvNN_Attn Spatial Samples K = 3
python allconvnet_main.py --layer ConvNN_Attn --K 3 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_K3

# # ConvNN_Attn Spatial Samples K = 4
python allconvnet_main.py --layer ConvNN_Attn --K 4 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_K4

# # ConvNN_Attn Spatial Samples K = 5
python allconvnet_main.py --layer ConvNN_Attn --K 5 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_K5  

# # ConvNN_Attn Spatial Samples K = 6
python allconvnet_main.py --layer ConvNN_Attn --K 6 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_K6

# # ConvNN_Attn Spatial Samples K = 7
python allconvnet_main.py --layer ConvNN_Attn --K 7 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_K7

# # ConvNN_Attn Spatial Samples K = 8
python allconvnet_main.py --layer ConvNN_Attn --K 8 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_K8

# # ConvNN_Attn Spatial Samples K = 9
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_K9

# # ConvNN_Attn Spatial Samples K = 10
python allconvnet_main.py --layer ConvNN_Attn --K 10 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10  --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_K10

# # ConvNN_Attn All Samples Coordinate Encoding K = 1
python allconvnet_main.py --layer ConvNN_Attn --K 1 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_Coord_K1 --coordinate_encoding   

# # ConvNN_Attn All Samples Coordinate Encoding K = 2
python allconvnet_main.py --layer ConvNN_Attn --K 2 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_Coord_K2 --coordinate_encoding

# # ConvNN_Attn All Samples Coordinate Encoding K = 3
python allconvnet_main.py --layer ConvNN_Attn --K 3 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_Coord_K3 --coordinate_encoding

# # ConvNN_Attn All Samples Coordinate Encoding K = 4
python allconvnet_main.py --layer ConvNN_Attn --K 4 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_Coord_K4 --coordinate_encoding

# # ConvNN_Attn All Samples Coordinate Encoding K = 5
python allconvnet_main.py --layer ConvNN_Attn --K 5 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_Coord_K5 --coordinate_encoding

# # ConvNN_Attn All Samples Coordinate Encoding K = 6
python allconvnet_main.py --layer ConvNN_Attn --K 6 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_Coord_K6 --coordinate_encoding

# # ConvNN_Attn All Samples Coordinate Encoding K = 7
python allconvnet_main.py --layer ConvNN_Attn --K 7 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_Coord_K7 --coordinate_encoding   

# # ConvNN_Attn All Samples Coordinate Encoding K = 8
python allconvnet_main.py --layer ConvNN_Attn --K 8 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_Coord_K8 --coordinate_encoding

# # ConvNN_Attn All Samples Coordinate Encoding K = 9
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_Coord_K9 --coordinate_encoding

# # ConvNN_Attn All Samples Coordinate Encoding K = 10
python allconvnet_main.py --layer ConvNN_Attn --K 10 --sampling_type all --num_layers 3 --channels 16 32 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_All_Coord_K10 --coordinate_encoding

# # ConvNN_Attn Random Samples Coordinate Encoding K = 1
python allconvnet_main.py --layer ConvNN_Attn --K 1 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_Coord_K1 --coordinate_encoding

# # ConvNN_Attn Random Samples Coordinate Encoding K = 2
python allconvnet_main.py --layer ConvNN_Attn --K 2 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_Coord_K2 --coordinate_encoding

# # ConvNN_Attn Random Samples Coordinate Encoding K = 3
python allconvnet_main.py --layer ConvNN_Attn --K 3 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_Coord_K3 --coordinate_encoding

# # ConvNN_Attn Random Samples Coordinate Encoding K = 4
python allconvnet_main.py --layer ConvNN_Attn --K 4 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_Coord_K4 --coordinate_encoding

# # ConvNN_Attn Random Samples Coordinate Encoding K = 5
python allconvnet_main.py --layer ConvNN_Attn --K 5 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_Coord_K5 --coordinate_encoding

# # ConvNN_Attn Random Samples Coordinate Encoding K = 6
python allconvnet_main.py --layer ConvNN_Attn --K 6 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_Coord_K6 --coordinate_encoding

# # ConvNN_Attn Random Samples Coordinate Encoding K = 7
python allconvnet_main.py --layer ConvNN_Attn --K 7 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_Coord_K7 --coordinate_encoding

# # ConvNN_Attn Random Samples Coordinate Encoding K = 8
python allconvnet_main.py --layer ConvNN_Attn --K 8 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_Coord_K8 --coordinate_encoding

# # ConvNN_Attn Random Samples Coordinate Encoding K = 9
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_Coord_K9

# # ConvNN_Attn Random Samples Coordinate Encoding K = 10
python allconvnet_main.py --layer ConvNN_Attn --K 10 --sampling_type random --num_layers 3 --channels 16 32 64 --num_samples 64 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Random_Coord_K10 --coordinate_encoding

# # ConvNN_Attn Spatial Samples Coordinate Encoding K = 1
python allconvnet_main.py --layer ConvNN_Attn --K 1 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_Coord_K1 --coordinate_encoding 

# # ConvNN_Attn Spatial Samples Coordinate Encoding K = 2
python allconvnet_main.py --layer ConvNN_Attn --K 2 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_Coord_K2 --coordinate_encoding

# # ConvNN_Attn Spatial Samples Coordinate Encoding K = 3
python allconvnet_main.py --layer ConvNN_Attn --K 3 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_Coord_K3 --coordinate_encoding

# # ConvNN_Attn Spatial Samples Coordinate Encoding K = 4
python allconvnet_main.py --layer ConvNN_Attn --K 4 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_Coord_K4 --coordinate_encoding

# # ConvNN_Attn Spatial Samples Coordinate Encoding K = 5
python allconvnet_main.py --layer ConvNN_Attn --K 5 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_Coord_K5 --coordinate_encoding

# # ConvNN_Attn Spatial Samples Coordinate Encoding K = 6
python allconvnet_main.py --layer ConvNN_Attn --K 6 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_Coord_K6 --coordinate_encoding

# # ConvNN_Attn Spatial Samples Coordinate Encoding K = 7
python allconvnet_main.py --layer ConvNN_Attn --K 7 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_Coord_K7 --coordinate_encoding

# # ConvNN_Attn Spatial Samples Coordinate Encoding K = 8
python allconvnet_main.py --layer ConvNN_Attn --K 8 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_Coord_K8 --coordinate_encoding

# # ConvNN_Attn Spatial Samples Coordinate Encoding K = 9
python allconvnet_main.py --layer ConvNN_Attn --K 9 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_Coord_K9 --coordinate_encoding

# # ConvNN_Attn Spatial Samples Coordinate Encoding K = 10
python allconvnet_main.py --layer ConvNN_Attn --K 10 --sampling_type spatial --num_layers 3 --channels 16 32 64 --num_samples 8 --dataset cifar10 --seed 0 --num_epochs 50 --device cuda --output_dir ./Output/Final_results/ACM/K_Test/ConvNN_Attn_Spatial_Coord_K10 --coordinate_encoding


### Study on Number of Samples 

#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=mnist1d-exp
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source activate mingi

### All Conv Net Experiments
python mnist1d_main.py --layer Conv1d --num_layers 4 --channels 64 32 16 8 --kernel_size 9 --num_epochs 200 --output_dir ./Output/Aug_25/1D/Conv1d_K9

python mnist1d_main.py --layer Conv1d --num_layers 4 --channels 64 32 16 8 --kernel_size 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/Conv1d_K8

python mnist1d_main.py --layer Conv1d --num_layers 4 --channels 64 32 16 8 --kernel_size 7 --num_epochs 200 --output_dir ./Output/Aug_25/1D/Conv1d_K7

python mnist1d_main.py --layer Conv1d --num_layers 4 --channels 64 32 16 8 --kernel_size 6 --num_epochs 200 --output_dir ./Output/Aug_25/1D/Conv1d_K6

python mnist1d_main.py --layer Conv1d --num_layers 4 --channels 64 32 16 8 --kernel_size 5 --num_epochs 200 --output_dir ./Output/Aug_25/1D/Conv1d_K5

python mnist1d_main.py --layer Conv1d --num_layers 4 --channels 64 32 16 8 --kernel_size 4 --num_epochs 200 --output_dir ./Output/Aug_25/1D/Conv1d_K4

python mnist1d_main.py --layer Conv1d --num_layers 4 --channels 64 32 16 8 --kernel_size 3 --num_epochs 200 --output_dir ./Output/Aug_25/1D/Conv1d_K3

python mnist1d_main.py --layer Conv1d --num_layers 4 --channels 64 32 16 8 --kernel_size 2 --num_epochs 200 --output_dir ./Output/Aug_25/1D/Conv1d_K2

python mnist1d_main.py --layer Conv1d --num_layers 4 --channels 64 32 16 8 --kernel_size 1 --num_epochs 200 --output_dir ./Output/Aug_25/1D/Conv1d_K1

# # With Coord ***NOT YET IMPLEMENTED
# python mnist1d_main.py --layer Conv2d_New --num_layers 4 --channels 64 32 16 8 --kernel_size 3 --num_epochs 200 --coordinate_encoding --output_dir ./Output/Aug_25/1D/Conv2d_New_K3_Coord

# python mnist1d_main.py --layer Conv2d_New --num_layers 4 --channels 64 32 16 8 --kernel_size 2 --num_epochs 200 --coordinate_encoding --output_dir ./Output/Aug_25/1D/Conv2d_New_K2_Coord

# python mnist1d_main.py --layer Conv2d_New --num_layers 4 --channels 64 32 16 8 --kernel_size 1 --num_epochs 200 --coordinate_encoding --output_dir ./Output/Aug_25/1D/Conv2d_New_K1_Coord

# ConvNN 
# K = 1
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 1 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K1

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 1 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K1

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 1 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K1

# K = 2
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 2 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K2

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 2 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K2

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 2 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K2

# K = 3
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 3 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K3

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 3 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K3

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 3 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K3

# K = 4
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 4 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K4

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 4 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K4

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 4 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K4

# K = 5 
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 5 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K5

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 5 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K5

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 5 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K5

# K = 6 
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 6 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K6

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 6 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K6

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 6 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K6

# K = 7 
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 7 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K7

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 7 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K7

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 7 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K7

# K = 8 
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 8 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K8

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 8 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K8

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 8 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K8

# K = 9 
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 9 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K9

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 9 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K9

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 9 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K9



# ConvNN with Coord
# K = 1
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 1 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K1_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 1 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K1_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 1 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K1_Coord --coordinate_encoding

# K = 2
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 2 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K2_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 2 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K2_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 2 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K2_Coord --coordinate_encoding

# K = 3
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 3 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K3_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 3 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K3_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 3 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K3_Coord --coordinate_encoding

# K = 4
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 4 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K4_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 4 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K4_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 4 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K4_Coord --coordinate_encoding

# K = 5 
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 5 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K5_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 5 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K5_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 5 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K5_Coord --coordinate_encoding

# K = 6 
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 6 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K6_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 6 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K6_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 6 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K6_Coord --coordinate_encoding

# K = 7 
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 7 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K7_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 7 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K7_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 7 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K7_Coord --coordinate_encoding

# K = 8 
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 8 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K8_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 8 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K8_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 8 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K8_Coord --coordinate_encoding

# K = 9 
python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 9 --sampling_type all --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_All_K9_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 9 --sampling_type random --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Random_K9_Coord --coordinate_encoding

python mnist1d_main.py --layer ConvNN --num_layers 4 --channels 64 32 16 8 --K 9 --sampling_type spatial --num_samples 8 --num_epochs 200 --output_dir ./Output/Aug_25/1D/ConvNN_Spatial_K9_Coord --coordinate_encoding

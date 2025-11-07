#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p arm --gres=shard:4
#SBATCH --cpus-per-task=12
#SBATCH --job-name=Install-NVIDIA-PyTorch
#SBATCH --time=00:30:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

# Setup conda
source ~/.bashrc
conda activate mingi-arm

conda env list

# Remove broken PIL/Pillow
pip uninstall PIL Pillow -y

# Clean install Pillow
pip install Pillow --force-reinstall

# echo "Installing NVIDIA PyTorch for ARM..."

# # Test it
# echo ""
# echo "Testing conda-installed PyTorch:"
# python -c "
# import torch
# print(f'PyTorch Version: {torch.__version__}')
# print(f'Built with CUDA: {torch.backends.cuda.is_built()}')
# print(f'CUDA Available: {torch.cuda.is_available()}')
# if torch.cuda.is_available():
#     print(f'GPU: {torch.cuda.get_device_name(0)}')
#     print('✅ SUCCESS!')
# else:
#     print('❌ Conda install failed, trying pip...')
# "
# echo "done!"



python main.py --num_epoch 50 --device cuda --output_dir ./Final_Output/arm_test/TEST
### Example shell script for Convolutional Nearest Neighbor



## 1. HPC Cluster Job A100 GPU 
#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=a100
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate mingi

## 2. HPC Cluster Job RTX 5090 GPU
#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=rtx-5090
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-rtx5090


## 3. HPC Cluster Job ARM GPU
#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p arm --gres=shard:4
#SBATCH --cpus-per-task=12
#SBATCH --job-name=ARM
#SBATCH --time=72:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate mingi-arm

## 4. Jetstream2 Cluster A100 GPU 
#!/bin/bash 
conda activate torch-a100



## **Jetstream2 Instructions**
# Jetstream2 Exouser Setup Script

# Note: 
# Jetstream2 instance is x86_64 architecture, not ARM architecture.

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh


source ~/.bashrc

conda --version

conda create -n torch-a100 python=3.11

conda activate torch-a100

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install numpy matplotlib Pillow thop tqdm einops torchsummary gpustat 

# Github Key Gen 
ssh-keygen -t ed25519 -C "mkang2@bowdoin.edu"

cat ~/.ssh/id_ed25519.pub

# should give the key and you add it to github ssh 

git clone git@github.com:mingikang31/Convolutional-Nearest-Neighbor.git

git clone git@github.com:mingikang31/Convolutional-Nearest-Neighbor-Attention.git


git config --global user.name "mingikang31"
git config --global user.email "mkang2@bowdoin.edu"

## SSH into Jetstream2 Instance
# ssh exouser@ip-address 
# password: Passphrase in Jetstream2 Instance page
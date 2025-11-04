#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=CVPR-Opt
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

source ~/.bashrc

conda activate mingi 





cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor

# ============ CONDA SETUP ============
# Auto-detect architecture and use appropriate conda
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    # Grace Hopper (ARM)
    source /mnt/research/j.farias/mkang2/miniconda3_arm/etc/profile.d/conda.sh
    echo "🔵 Using ARM conda (Grace Hopper)"
else
    # Bowdoin A100 (x86-64)
    source /mnt/research/j.farias/mkang2/miniconda3/etc/profile.d/conda.sh
    echo "🟢 Using x86-64 conda (Bowdoin A100)"
fi

conda activate mingi

echo "Architecture: $ARCH"
echo "Python: $(python --version)"
echo ""

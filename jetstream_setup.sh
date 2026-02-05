### JetStream2 H100 g5.4xl setup script ###

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh


source ~/.bashrc

conda --version

conda create -n torch-h100 python=3.11

conda activate torch-h100

pip install torch torchvision torchaudio datasets transformers tokenizers pytorch-ignite pytorch-lightning matplotlib numpy Pillow einops torchsummary gpustat

# Github Key Gen
ssh-keygen -t ed25519 -C "mkang2@bowdoin.edu"
cat ~/.ssh/id_ed25519.pub

# Git clone repos
git clone git@github.com:mingikang31/Convolutional-Nearest-Neighbor.git
git clone git@github.com:mingikang31/Convolutional-Nearest-Neighbor-Attention.git

git config --global user.name "mingikang31"
git config --global user.email "mkang2@bowdoin.edu"
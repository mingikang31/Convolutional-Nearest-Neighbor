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


# Send completion email
echo "K-Test experiments completed!
Total: $TOTAL
Successful: $((TOTAL - FAILED))
Failed: $FAILED

Check the output directory for results." | mail -s "K-Test Experiments Done" mkang2@bowdoin.edu

# ssh exouser@ip-address 
# password: Passphrase in Jetstream2 Instance
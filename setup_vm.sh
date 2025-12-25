#!/bin/bash
# One-command VM setup for 2x A100 training
# Run: bash setup_vm.sh

set -e

echo "=== Setting up VM for multi-GPU training ==="

# 1. Mount local SSD
echo "[1/5] Mounting local SSD..."
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/ssd
sudo mount /dev/nvme0n1 /mnt/ssd
sudo chown $USER:$USER /mnt/ssd

# 2. Set HuggingFace cache
echo "[2/5] Configuring HuggingFace cache..."
export HF_HOME=/mnt/ssd/hf_cache
mkdir -p $HF_HOME
echo 'export HF_HOME=/mnt/ssd/hf_cache' >> ~/.bashrc

# 3. Install Python venv (skip conda entirely)
echo "[3/5] Setting up Python..."
sudo apt update
sudo apt install -y python3-pip python3-venv

# 4. Create venv and install deps
echo "[4/5] Installing dependencies..."
python3 -m venv /mnt/ssd/venv
source /mnt/ssd/venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate

# 5. Copy project to SSD
echo "[5/5] Moving project to SSD..."
cp -r ~/Gptt /mnt/ssd/Gptt

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To train, run:"
echo "  source /mnt/ssd/venv/bin/activate"
echo "  cd /mnt/ssd/Gptt"
echo "  accelerate launch train.py"
echo ""
echo "Use tmux to keep training alive if you disconnect:"
echo "  tmux new -s training"

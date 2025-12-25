#!/bin/bash
# Single script to setup and train on GCE VM
# Usage: bash run.sh

set -e

echo "=== Charlie Kirk LoRA Training ==="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    echo "[2/4] Installing dependencies..."
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets peft accelerate sentencepiece protobuf
else
    echo "[1/4] Virtual environment exists, activating..."
    source venv/bin/activate
    echo "[2/4] Dependencies already installed"
fi

echo ""
echo "[3/4] Preprocessing training data..."
python3 preprocess.py

echo ""
echo "[4/4] Starting training..."
python3 train.py

echo ""
echo "=== Training complete! ==="
echo "Adapters saved to: output/adapters/"
echo ""
echo "To download, use Cloud Shell:"
echo "  gcloud compute scp VM_NAME:~/Gptt/output/adapters ./adapters --recurse --zone=ZONE"

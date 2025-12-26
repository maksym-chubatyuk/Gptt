#!/bin/bash
# Single script to setup, train, and convert to GGUF on GCE VM
# Usage: bash run.sh

set -e

echo "=== Charlie Kirk LoRA Training + GGUF Conversion ==="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "[1/7] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    echo "[2/7] Installing dependencies..."
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets peft accelerate sentencepiece protobuf
else
    echo "[1/7] Virtual environment exists, activating..."
    source venv/bin/activate
    echo "[2/7] Dependencies already installed"
fi

echo ""
echo "[3/7] Preprocessing training data (includes vision examples)..."
python3 preprocess.py

echo ""
echo "[4/7] Starting training..."
python3 train.py

echo ""
echo "[5/7] Setting up llama.cpp for GGUF conversion..."
if [ ! -d "llama.cpp" ]; then
    echo "  Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
    pip install -r llama.cpp/requirements.txt
else
    echo "  llama.cpp already exists"
fi

echo ""
echo "[6/7] Building llama.cpp quantize tool..."
cd llama.cpp
if [ ! -f "quantize" ] && [ ! -f "llama-quantize" ] && [ ! -f "build/bin/llama-quantize" ]; then
    echo "  Building quantize..."
    make -j quantize
else
    echo "  quantize already built"
fi
cd ..

echo ""
echo "[7/7] Merging adapters and converting to GGUF..."
python3 merge_and_convert.py

echo ""
echo "=== Training & Conversion Complete! ==="
echo ""
echo "Output files:"
echo "  - LoRA adapters: output/adapters/"
echo "  - GGUF model: output/model.gguf"
echo ""
echo "To upload GGUF model to GCS:"
echo "  bash upload.sh"
echo ""
echo "To download to your local machine:"
echo "  gsutil cp gs://maksym-adapters/model.gguf ~/Desktop/AI/Gptt/output/"

#!/bin/bash
# Setup script for inference on Mac (Metal) or Linux (CUDA)
# Creates output folder, builds llama.cpp, installs dependencies
# Optionally removes training files to keep only inference code

set -e

echo "=============================================="
echo "  Qwen3-VL Chat Setup"
echo "=============================================="
echo ""

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Darwin*)    PLATFORM="mac" ;;
    Linux*)     PLATFORM="linux" ;;
    *)          echo "Unsupported OS: $OS"; exit 1 ;;
esac

echo "Detected platform: $PLATFORM"
echo ""

# Create output folder
echo "[1/4] Creating output folder..."
mkdir -p output
echo "  Created: output/"
echo ""

# Create venv and install Python dependencies
echo "[2/4] Setting up Python environment..."
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
fi

echo "  Activating venv and installing dependencies..."
source venv/bin/activate
pip install -q --upgrade pip
pip install -q opencv-python Pillow requests
echo "  Installed: opencv-python, Pillow, requests"
echo ""

# Clone and build llama.cpp
echo "[3/4] Setting up llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    echo "  Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
else
    echo "  llama.cpp exists, updating..."
    cd llama.cpp && git pull && cd ..
fi

echo "  Building llama-server..."
cd llama.cpp

if [ "$PLATFORM" = "mac" ]; then
    echo "  Configuring with Metal backend..."
    cmake -B build -DGGML_METAL=on -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
else
    echo "  Configuring with CUDA backend..."
    cmake -B build -DGGML_CUDA=on -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
fi

cmake --build build --target llama-server -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) > /dev/null 2>&1
cd ..

if [ -f "llama.cpp/build/bin/llama-server" ]; then
    echo "  Build successful!"
else
    echo "  Error: Build failed. Check llama.cpp/build for errors."
    exit 1
fi
echo ""

# Ask about cleaning training files
echo "[4/4] Cleanup..."
echo ""
echo "The following files are only needed for training and can be removed:"
echo "  - train.py"
echo "  - preprocess.py"
echo "  - merge_and_convert.py"
echo "  - requirements-training.txt"
echo "  - run.sh"
echo "  - upload.sh"
echo "  - data/ folder"
echo "  - output/adapters/"
echo "  - output/merged_fp16/"
echo ""

read -p "Remove training files? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Removing training files..."
    rm -f train.py preprocess.py merge_and_convert.py
    rm -f requirements-training.txt run.sh upload.sh
    rm -rf data/
    rm -rf output/adapters/ output/merged_fp16/
    echo "  Done!"
else
    echo "  Keeping training files."
fi

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download your models into the output/ folder:"
echo "   - output/model.gguf (required, ~5GB)"
echo "   - output/mmproj-model.gguf (for vision, ~1.1GB)"
echo ""
echo "   From GCS:"
echo "     gsutil cp gs://maksym-adapters/model.gguf output/"
echo ""
echo "   Or download base vision projector from HuggingFace:"
echo "     curl -L -o output/mmproj-model.gguf \\"
echo "       'https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf'"
echo ""
echo "2. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "3. Run the chat:"
echo "     python3 main.py"
echo ""

#!/bin/bash
# Upload GGUF model to GCS
# Usage: bash upload.sh

set -e

BUCKET="gs://maksym-adapters"

echo "=== Uploading GGUF model to GCS ==="

# Check if GGUF model exists
if [ ! -f "output/model.gguf" ]; then
    echo "Error: No model.gguf found"
    echo "Run training and conversion first: bash run.sh"
    exit 1
fi

# Create bucket if it doesn't exist (ignore error if exists)
gsutil mb $BUCKET 2>/dev/null || true

# Upload GGUF model
echo "Uploading model.gguf (this may take a while, ~4GB)..."
gsutil cp output/model.gguf $BUCKET/

# Also upload adapters as backup (optional)
if [ -f "output/adapters/adapter_model.safetensors" ]; then
    echo "Uploading adapters as backup..."
    gsutil cp output/adapters/adapter_model.safetensors $BUCKET/adapters/
    gsutil cp output/adapters/adapter_config.json $BUCKET/adapters/
fi

echo ""
echo "=== Done! ==="
echo "Download from: https://console.cloud.google.com/storage/browser/maksym-adapters"
echo ""
echo "On your Mac, download the GGUF model:"
echo "  mkdir -p ~/Desktop/AI/Gptt/output"
echo "  gsutil cp gs://maksym-adapters/model.gguf ~/Desktop/AI/Gptt/output/"
echo ""
echo "Then download LLaVA vision model:"
echo "  mkdir -p ~/Desktop/AI/Gptt/models"
echo "  wget -P ~/Desktop/AI/Gptt/models https://huggingface.co/cjpais/llava-v1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf"
echo "  wget -P ~/Desktop/AI/Gptt/models https://huggingface.co/cjpais/llava-v1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf"
echo ""
echo "Finally, run the vision-enabled chat:"
echo "  python main_vision.py"

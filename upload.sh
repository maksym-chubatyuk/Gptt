#!/bin/bash
# Upload GGUF models to GCS
# For Qwen3-VL-8B model (main model + vision projector)
# Usage: bash upload.sh

set -e

BUCKET="gs://maksym-adapters"

echo "=== Uploading GGUF models to GCS ==="

# Check if main GGUF model exists
if [ ! -f "output/model.gguf" ]; then
    echo "Error: No model.gguf found"
    echo "Run training and conversion first: bash run.sh"
    exit 1
fi

# Create bucket if it doesn't exist (ignore error if exists)
gsutil mb $BUCKET 2>/dev/null || true

# Upload main GGUF model
echo "Uploading model.gguf (~5GB)..."
gsutil cp output/model.gguf $BUCKET/

# Upload vision projector if it exists
if [ -f "output/mmproj-model.gguf" ]; then
    echo "Uploading mmproj-model.gguf (~600MB)..."
    gsutil cp output/mmproj-model.gguf $BUCKET/
fi

echo ""
echo "=== Done! ==="
echo ""
echo "On your PC, download and run:"
echo "  mkdir -p output"
echo "  gsutil cp gs://maksym-adapters/model.gguf output/"
echo "  gsutil cp gs://maksym-adapters/mmproj-model.gguf output/"
echo "  python main.py"

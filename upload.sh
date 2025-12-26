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
echo "Uploading model.gguf (~4GB)..."
gsutil cp output/model.gguf $BUCKET/

echo ""
echo "=== Done! ==="
echo ""
echo "On your PC, download and run:"
echo "  gsutil cp gs://maksym-adapters/model.gguf output/"
echo "  python main.py"

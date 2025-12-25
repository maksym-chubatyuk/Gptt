#!/bin/bash
# Upload trained adapters to GCS
# Usage: bash upload.sh

set -e

BUCKET="gs://maksym-adapters"

echo "=== Uploading adapters to GCS ==="

# Check if adapters exist
if [ ! -f "output/adapters/adapter_model.safetensors" ]; then
    echo "Error: No adapter_model.safetensors found"
    echo "Run training first: bash run.sh"
    exit 1
fi

# Create bucket if it doesn't exist (ignore error if exists)
gsutil mb $BUCKET 2>/dev/null || true

# Upload files
echo "Uploading adapter_model.safetensors..."
gsutil cp output/adapters/adapter_model.safetensors $BUCKET/

echo "Uploading adapter_config.json..."
gsutil cp output/adapters/adapter_config.json $BUCKET/

echo ""
echo "=== Done! ==="
echo "Download from: https://console.cloud.google.com/storage/browser/maksym-adapters"
echo ""
echo "Or on your Mac:"
echo "  gsutil cp gs://maksym-adapters/adapter_model.safetensors ~/Desktop/AI/Gptt/output/adapters/"
echo "  gsutil cp gs://maksym-adapters/adapter_config.json ~/Desktop/AI/Gptt/output/adapters/"

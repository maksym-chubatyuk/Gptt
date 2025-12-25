#!/bin/bash
# Push trained adapters to GitHub from GCE VM

set -e

echo "=== Pushing trained adapters to GitHub ==="

# Check if adapters exist
if [ ! -f "output/adapters/adapter_model.safetensors" ]; then
    echo "Error: No trained adapters found in output/adapters/"
    echo "Run training first: accelerate launch train.py"
    exit 1
fi

# Stage adapter files (force add since they're in .gitignore)
git add -f output/adapters/adapter_model.safetensors
git add -f output/adapters/adapter_config.json
git add -f output/adapters/tokenizer*
git add -f output/adapters/special_tokens_map.json

# Commit
git commit -m "Trained adapters $(date +%Y-%m-%d)"

# Push (will prompt for credentials)
git push

echo ""
echo "=== Done! Adapters pushed to GitHub ==="

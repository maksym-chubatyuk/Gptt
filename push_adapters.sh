#!/bin/bash
# Push trained adapters to GitHub from GCE VM

set -e

echo "=== Pushing trained adapters to GitHub ==="
echo ""

# Check if adapters exist
echo "[1/5] Checking for trained adapters..."
if [ ! -f "output/adapters/adapter_model.safetensors" ]; then
    echo "  ERROR: No trained adapters found in output/adapters/"
    echo "  Run training first: accelerate launch train.py"
    exit 1
fi
echo "  Found adapter_model.safetensors ($(du -h output/adapters/adapter_model.safetensors | cut -f1))"
echo ""

# Stage adapter files (force add since they're in .gitignore)
echo "[2/5] Staging adapter files..."
echo "  Adding adapter_model.safetensors..."
git add -f output/adapters/adapter_model.safetensors
echo "  Adding adapter_config.json..."
git add -f output/adapters/adapter_config.json
echo "  Adding tokenizer files..."
git add -f output/adapters/tokenizer*
echo "  Adding special_tokens_map.json..."
git add -f output/adapters/special_tokens_map.json
echo "  Done staging files"
echo ""

# Show what's staged
echo "[3/5] Files to be committed:"
git diff --cached --stat
echo ""

# Commit
echo "[4/5] Creating commit..."
git commit -m "Trained adapters $(date +%Y-%m-%d)"
echo ""

# Push (will prompt for credentials)
echo "[5/5] Pushing to GitHub (this may take a few minutes)..."
echo "  Uploading ~80MB adapter file..."
git push --progress

echo ""
echo "=== Done! Adapters pushed to GitHub ==="

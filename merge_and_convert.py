#!/usr/bin/env python3
"""
Merge LoRA adapters into base model and convert to GGUF format.
Run this after training to create a standalone 4-bit quantized model.

Requirements:
- Trained LoRA adapters in output/adapters/
- llama.cpp repository cloned (for conversion script)
"""

import os
import sys
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "output/adapters"
MERGED_PATH = "output/merged_fp16"
GGUF_PATH = "output/model.gguf"
LLAMA_CPP_PATH = "llama.cpp"


def check_prerequisites():
    """Check that all prerequisites are met."""
    # Check for adapters
    adapter_path = Path(ADAPTER_PATH)
    if not adapter_path.exists():
        print(f"Error: Adapter path {ADAPTER_PATH} does not exist!")
        print("Run training first: python train.py")
        return False

    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        print(f"Error: No adapter_config.json found in {ADAPTER_PATH}")
        return False

    # Check for llama.cpp
    llama_cpp = Path(LLAMA_CPP_PATH)
    if not llama_cpp.exists():
        print(f"Error: llama.cpp not found at {LLAMA_CPP_PATH}")
        print("Clone it with: git clone https://github.com/ggerganov/llama.cpp")
        return False

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"Error: convert_hf_to_gguf.py not found in {LLAMA_CPP_PATH}")
        print("Make sure you have the latest llama.cpp")
        return False

    return True


def merge_adapters():
    """Merge LoRA adapters into the base model."""
    print("=" * 50)
    print("Step 1: Merging LoRA Adapters")
    print("=" * 50)

    # Check CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. This will be slow on CPU.")

    print(f"\nLoading base model: {MODEL}")
    print("  This may take a few minutes...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"\nLoading adapters from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("\nMerging adapters into base model...")
    model = model.merge_and_unload()

    # Create output directory
    os.makedirs(MERGED_PATH, exist_ok=True)

    print(f"\nSaving merged model to: {MERGED_PATH}")
    model.save_pretrained(MERGED_PATH)

    # Also save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.save_pretrained(MERGED_PATH)

    print("\nMerge complete!")

    # Print memory usage if CUDA available
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory used: {memory_gb:.2f} GB")


def convert_to_gguf():
    """Convert merged HuggingFace model to GGUF format (two-step: convert + quantize)."""
    print("\n" + "=" * 50)
    print("Step 2: Converting to GGUF (f16)")
    print("=" * 50)

    convert_script = Path(LLAMA_CPP_PATH) / "convert_hf_to_gguf.py"
    f16_gguf = "output/model-f16.gguf"

    # Create output directory
    output_dir = Path(GGUF_PATH).parent
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nInput: {MERGED_PATH}")
    print(f"Output (f16): {f16_gguf}")
    print("\nRunning conversion to f16 GGUF...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(convert_script),
                MERGED_PATH,
                "--outfile", f16_gguf,
                "--outtype", "f16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return False

    # Check f16 output exists
    if not Path(f16_gguf).exists():
        print(f"Error: f16 GGUF not created at {f16_gguf}")
        return False

    print(f"\nf16 GGUF created: {f16_gguf}")

    # Step 2b: Quantize to Q4_K_M
    print("\n" + "=" * 50)
    print("Step 3: Quantizing to Q4_K_M (4-bit)")
    print("=" * 50)

    # Find llama-quantize binary (CMake build location)
    quantize_bin = Path(LLAMA_CPP_PATH) / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        # Try to build it with CMake
        print("llama-quantize not found. Attempting to build with CMake...")
        try:
            subprocess.run(
                ["cmake", "-B", "build"],
                cwd=LLAMA_CPP_PATH,
                check=True,
            )
            subprocess.run(
                ["cmake", "--build", "build", "--target", "llama-quantize", "-j"],
                cwd=LLAMA_CPP_PATH,
                check=True,
            )
            quantize_bin = Path(LLAMA_CPP_PATH) / "build" / "bin" / "llama-quantize"
        except subprocess.CalledProcessError:
            print("Error: Could not build llama-quantize.")
            print("The f16 GGUF is ready, but not quantized.")
            print(f"You can manually quantize with:")
            print(f"  cd {LLAMA_CPP_PATH}")
            print(f"  cmake -B build && cmake --build build --target llama-quantize -j")
            print(f"  ./build/bin/llama-quantize {f16_gguf} {GGUF_PATH} Q4_K_M")
            # Copy f16 as fallback
            import shutil
            shutil.copy(f16_gguf, GGUF_PATH)
            return True

    print(f"\nQuantizing: {f16_gguf} -> {GGUF_PATH}")
    print("Quantization type: Q4_K_M")

    try:
        result = subprocess.run(
            [str(quantize_bin), f16_gguf, GGUF_PATH, "Q4_K_M"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error during quantization: {e}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        print(f"\nFalling back to f16 GGUF (larger but usable)")
        import shutil
        shutil.copy(f16_gguf, GGUF_PATH)

    # Clean up f16 intermediate file
    f16_path = Path(f16_gguf)
    if f16_path.exists() and Path(GGUF_PATH).exists():
        f16_size = f16_path.stat().st_size / (1024**3)
        print(f"\nCleaning up intermediate f16 GGUF ({f16_size:.2f} GB)...")
        f16_path.unlink()

    # Check final output
    gguf_file = Path(GGUF_PATH)
    if gguf_file.exists():
        size_gb = gguf_file.stat().st_size / (1024**3)
        print(f"\nGGUF model created successfully!")
        print(f"  Path: {GGUF_PATH}")
        print(f"  Size: {size_gb:.2f} GB")
        return True
    else:
        print(f"Error: GGUF file not created at {GGUF_PATH}")
        return False


def cleanup_intermediate():
    """Optionally clean up intermediate merged_fp16 folder."""
    import shutil

    merged_path = Path(MERGED_PATH)
    if merged_path.exists():
        size_gb = sum(f.stat().st_size for f in merged_path.rglob('*') if f.is_file()) / (1024**3)
        print(f"\nIntermediate files at {MERGED_PATH}: {size_gb:.2f} GB")
        print("You can delete this folder to save space once conversion is complete.")
        print(f"  rm -rf {MERGED_PATH}")


def main():
    print("=" * 50)
    print("  LoRA Merge & GGUF Conversion")
    print("=" * 50)
    print()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Step 1: Merge adapters
    merge_adapters()

    # Step 2: Convert to GGUF
    success = convert_to_gguf()

    if success:
        print("\n" + "=" * 50)
        print("  Conversion Complete!")
        print("=" * 50)
        print(f"\nYour GGUF model is ready at: {GGUF_PATH}")
        print("\nTo use it locally:")
        print("  1. Install llama-cpp-python:")
        print('     CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python')
        print("  2. Run: python main_vision.py")
        print("\nTo upload to GCS:")
        print("  bash upload.sh")

        cleanup_intermediate()
    else:
        print("\nConversion failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

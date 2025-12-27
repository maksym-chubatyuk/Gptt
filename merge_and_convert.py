#!/usr/bin/env python3
"""
Merge LoRA adapters into base model and convert to GGUF format.
Run this after training to create a standalone 4-bit quantized model.

For Qwen3-VL, this produces two GGUF files:
- model.gguf: Main language model (quantized to Q4_K_M)
- mmproj-model.gguf: Vision projector (kept at F16 for quality)

Requirements:
- Trained LoRA adapters in output/adapters/
- llama.cpp repository cloned (for conversion script)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

# Configuration
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER_PATH = "output/adapters"
MERGED_PATH = "output/merged_fp16"
GGUF_PATH = "output/model.gguf"
MMPROJ_PATH = "output/mmproj-model.gguf"
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

    base_model = AutoModelForVision2Seq.from_pretrained(
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

    # Also save processor (includes tokenizer)
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    processor.save_pretrained(MERGED_PATH)

    print("\nMerge complete!")

    # Print memory usage if CUDA available
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory used: {memory_gb:.2f} GB")


def convert_to_gguf():
    """Convert merged HuggingFace model to GGUF format.

    For Qwen3-VL, this produces two files:
    - Main model GGUF (quantized to Q4_K_M)
    - Vision projector GGUF (kept at F16)
    """
    print("\n" + "=" * 50)
    print("Step 2: Converting to GGUF (f16)")
    print("=" * 50)

    convert_script = Path(LLAMA_CPP_PATH) / "convert_hf_to_gguf.py"
    f16_gguf = "output/model-f16.gguf"
    f16_mmproj = "output/mmproj-model-f16.gguf"

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

    # Check for vision projector (mmproj) - Qwen3-VL creates this automatically
    # Look for mmproj file in output directory
    mmproj_candidates = list(Path("output").glob("*mmproj*.gguf"))
    if mmproj_candidates:
        mmproj_src = mmproj_candidates[0]
        print(f"\nVision projector found: {mmproj_src}")
        if mmproj_src != Path(MMPROJ_PATH):
            shutil.copy(mmproj_src, MMPROJ_PATH)
            print(f"  Copied to: {MMPROJ_PATH}")
    else:
        print("\nNote: No separate mmproj file found.")
        print("  This is normal if the vision encoder is integrated into the main model.")

    # Step 2b: Quantize main model to Q4_K_M
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
        shutil.copy(f16_gguf, GGUF_PATH)

    # Clean up f16 intermediate files
    f16_path = Path(f16_gguf)
    if f16_path.exists() and Path(GGUF_PATH).exists():
        f16_size = f16_path.stat().st_size / (1024**3)
        print(f"\nCleaning up intermediate f16 GGUF ({f16_size:.2f} GB)...")
        f16_path.unlink()

    # Clean up intermediate mmproj if we copied it
    for candidate in mmproj_candidates:
        if candidate.exists() and candidate != Path(MMPROJ_PATH):
            candidate.unlink()

    # Check final output
    gguf_file = Path(GGUF_PATH)
    mmproj_file = Path(MMPROJ_PATH)

    if gguf_file.exists():
        size_gb = gguf_file.stat().st_size / (1024**3)
        print(f"\nGGUF model created successfully!")
        print(f"  Main model: {GGUF_PATH} ({size_gb:.2f} GB)")

        if mmproj_file.exists():
            mmproj_size_mb = mmproj_file.stat().st_size / (1024**2)
            print(f"  Vision projector: {MMPROJ_PATH} ({mmproj_size_mb:.0f} MB)")

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
        print(f"\nYour GGUF files are ready:")
        print(f"  - Main model: {GGUF_PATH}")
        if Path(MMPROJ_PATH).exists():
            print(f"  - Vision projector: {MMPROJ_PATH}")
        print("\nTo use it locally:")
        print("  1. Install llama-cpp-python:")
        print('     CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python')
        print("  2. Run: python main.py")
        print("\nTo upload to GCS:")
        print("  bash upload.sh")

        cleanup_intermediate()
    else:
        print("\nConversion failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

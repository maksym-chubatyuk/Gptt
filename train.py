#!/usr/bin/env python3
"""
Fine-tune a language model using PyTorch + PEFT LoRA.
Trains on data/train.jsonl and saves adapters to output/adapters.
Supports NVIDIA GPUs via CUDA with 4-bit quantization (QLoRA).
"""

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


class DataCollatorForCausalLM:
    """Custom data collator that properly pads input_ids, attention_mask, and labels."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Find max length in batch
        max_length = max(len(f["input_ids"]) for f in features)

        # Round up to multiple for efficiency
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1)
                          // self.pad_to_multiple_of * self.pad_to_multiple_of)

        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            seq_len = len(input_ids)
            padding_len = max_length - seq_len

            # Pad input_ids with pad_token_id
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask = [1] * seq_len + [0] * padding_len
            # Labels: -100 for padding (ignored in loss)
            padded_labels = labels + [-100] * padding_len

            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(padded_labels)

        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch

# Configuration
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_DIR = "data"
OUTPUT_DIR = "output/adapters"

# Training hyperparameters (matching original MLX config)
MAX_STEPS = 1200  # Equivalent to ITERS in MLX
BATCH_SIZE = 2  # Reduced for memory; use gradient accumulation to compensate
LEARNING_RATE = 5e-6
GRADIENT_ACCUMULATION_STEPS = 1  # Effective batch size = 2 GPUs * 2 batch * 1 = 4
WARMUP_STEPS = 50
LOGGING_STEPS = 10
EVAL_STEPS = 50  # More frequent evaluation (was STEPS_PER_EVAL)
SAVE_STEPS = 100  # More frequent checkpoints (was SAVE_EVERY)
MAX_SEQ_LENGTH = 2048

# LoRA Configuration
LORA_R = 8  # rank
LORA_ALPHA = 16  # scale factor
LORA_DROPOUT = 0.0

# Target modules for Mistral architecture
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]



def check_data():
    """Verify training data exists."""
    train_file = Path(DATA_DIR) / "train.jsonl"
    valid_file = Path(DATA_DIR) / "valid.jsonl"

    if not train_file.exists():
        print(f"Error: {train_file} not found!")
        print("Run 'python preprocess.py' first to generate training data.")
        return False

    if not valid_file.exists():
        print(f"Warning: {valid_file} not found. Validation will be skipped.")

    # Count examples
    with open(train_file) as f:
        train_count = sum(1 for _ in f)
    print(f"Training examples: {train_count}")

    return True


def load_training_data(tokenizer):
    """Load and preprocess training data from JSONL files."""
    train_file = Path(DATA_DIR) / "train.jsonl"
    valid_file = Path(DATA_DIR) / "valid.jsonl"

    # Load datasets
    data_files = {"train": str(train_file)}
    if valid_file.exists():
        data_files["validation"] = str(valid_file)

    dataset = load_dataset("json", data_files=data_files)

    def format_and_tokenize(example):
        """Format a single example using chat template and tokenize."""
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        result = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    # Process datasets
    dataset = dataset.map(
        format_and_tokenize,
        remove_columns=["messages"],
        desc="Tokenizing"
    )

    return dataset


def train():
    """Run LoRA fine-tuning with PEFT."""
    print("=" * 50)
    print("PyTorch + PEFT LoRA Fine-tuning")
    print("=" * 50)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nError: CUDA not available. Training requires a GPU.")
        print("Make sure you have:")
        print("  1. An NVIDIA GPU")
        print("  2. CUDA toolkit installed")
        print("  3. PyTorch installed with CUDA support")
        sys.exit(1)

    # Check prerequisites
    if not check_data():
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    print(f"\nGPUs available: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Model: {MODEL}")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"LoRA rank: {LORA_R}")
    print(f"Learning rate: {LEARNING_RATE}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load and preprocess data
    print("Loading training data...")
    dataset = load_training_data(tokenizer)
    print(f"  Training examples: {len(dataset['train'])}")
    if "validation" in dataset:
        print(f"  Validation examples: {len(dataset['validation'])}")

    # Load base model in fp16
    print("\nLoading base model in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # More memory efficient attention
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps" if "validation" in dataset else "no",
        eval_steps=EVAL_STEPS if "validation" in dataset else None,
        save_steps=SAVE_STEPS,
        save_total_limit=5,
        fp16=True,  # Use mixed precision
        optim="adamw_torch",
        lr_scheduler_type="linear",
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,  # Save memory
    )

    # Data collator
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=data_collator,
    )

    # Start training
    print("\n" + "-" * 50)
    print("Starting training...")
    print("-" * 50 + "\n")

    try:
        trainer.train()

        # Save final adapters
        print("\nSaving adapters...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print("\n" + "=" * 50)
        print("Training complete!")
        print(f"Adapters saved to: {OUTPUT_DIR}")
        print("=" * 50)
        print("\nTo chat with your fine-tuned model, run:")
        print("  python main.py")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Partial adapters may be saved in: {OUTPUT_DIR}")
        sys.exit(0)


if __name__ == "__main__":
    train()

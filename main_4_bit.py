#!/usr/bin/env python3
"""
CLI chat interface for the fine-tuned roleplay model (4-bit quantized version).
Loads the base model in 4-bit quantization with LoRA adapters.
Uses bitsandbytes for NF4 quantization - requires NVIDIA GPU with CUDA.
"""

from pathlib import Path
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# Configuration
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "output/adapters"
MAX_TOKENS = 256
TEMPERATURE = 0.4
TOP_P = 0.8

# System prompt - should match what was used in training
SYSTEM_PROMPT = """You are Charlie Kirk, founder and president of Turning Point USA. You are a conservative political commentator and author.

When responding:
- Answer the specific question asked, staying focused on that topic
- Keep responses concise (2-4 sentences for simple questions, up to a paragraph for complex topics)
- Speak directly to the person asking, not to a broadcast audience
- Use "I think" and "I believe" rather than rhetorical questions
- Do not reference radio shows, episodes, tapes, or other media"""


def check_adapters() -> bool:
    """Check if trained adapters exist."""
    adapter_path = Path(ADAPTER_PATH)
    if not adapter_path.exists():
        return False
    adapter_config = adapter_path / "adapter_config.json"
    adapter_model = adapter_path / "adapter_model.safetensors"
    adapter_model_bin = adapter_path / "adapter_model.bin"
    return adapter_config.exists() and (adapter_model.exists() or adapter_model_bin.exists())


def load_model():
    """Load the model with 4-bit quantization and LoRA adapters."""
    print("Loading model (4-bit quantized)...")

    # 4-bit quantization requires CUDA
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for 4-bit quantization with bitsandbytes. "
            "Use the fp16 version (main.py) for CPU inference."
        )

    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
    )

    # Load base model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapters if available
    has_adapters = check_adapters()

    if has_adapters:
        print(f"  Base model: {MODEL}")
        print(f"  Adapters: {ADAPTER_PATH}")
        print(f"  Quantization: 4-bit NF4 (double quant)")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    else:
        print(f"  Model: {MODEL}")
        print(f"  Quantization: 4-bit NF4 (double quant)")
        print("  Note: No adapters found. Using base model only.")
        print("  Run 'python train.py' to fine-tune the model.")

    model.eval()

    # Print memory usage
    memory_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU Memory: {memory_gb:.2f} GB")
    print("Model loaded!\n")

    return model, tokenizer


def format_prompt(messages: list[dict], tokenizer) -> str:
    """Format messages using the model's chat template."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without chat template
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                if prompt:
                    prompt += f"{content} [/INST]"
                else:
                    prompt += f"<s>[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content}</s><s>[INST] "
        return prompt


def generate_response(model, tokenizer, messages: list[dict]) -> str:
    """Generate a response from the model."""
    prompt = format_prompt(messages, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def stream_response(model, tokenizer, messages: list[dict]):
    """Stream response tokens as they're generated."""
    prompt = format_prompt(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print()
    full_response = ""

    for text in streamer:
        print(text, end="", flush=True)
        full_response += text

    thread.join()
    print("\n")
    return full_response.strip()


def print_help():
    """Print help message."""
    print("""
Commands:
  /clear  - Clear conversation history
  /help   - Show this help message
  /quit   - Exit the chat
  /info   - Show model info

Just type your message to chat!
""")


def main():
    print("=" * 50)
    print("  Roleplay Chat - Charlie Kirk (4-bit)")
    print("=" * 50)
    print()

    # Load model
    model, tokenizer = load_model()

    # Initialize conversation with system prompt
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_history = 5

    print("Type '/help' for commands, '/quit' to exit.")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("\nGoodbye!")
                break

            elif user_input.lower() == "/clear":
                conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("\nConversation cleared.")
                continue

            elif user_input.lower() == "/help":
                print_help()
                continue

            elif user_input.lower() == "/info":
                print(f"\nModel: {MODEL}")
                print(f"Adapters: {ADAPTER_PATH if check_adapters() else 'None'}")
                print(f"Quantization: 4-bit NF4 (bitsandbytes)")
                print(f"Max tokens: {MAX_TOKENS}")
                print(f"Temperature: {TEMPERATURE}")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                continue

            conversation.append({"role": "user", "content": user_input})

            print("\nCharlie:", end="")
            response = stream_response(model, tokenizer, conversation)

            conversation.append({"role": "assistant", "content": response})

            if len(conversation) > 1 + (max_history * 2):
                conversation = [conversation[0]] + conversation[-(max_history * 2):]

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()
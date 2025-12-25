#!/usr/bin/env python3
"""
CLI chat interface for the fine-tuned roleplay model.
Loads the base model with LoRA adapters and provides an interactive chat experience.
Uses PyTorch + Transformers + PEFT for NVIDIA GPU support.
"""

from pathlib import Path
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from peft import PeftModel

# Configuration
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "output/adapters"
MAX_TOKENS = 256  # Reduced from 512 to encourage concise responses
TEMPERATURE = 0.4  # Reduced from 0.7 for more focused responses
TOP_P = 0.8  # Reduced from 0.9 to exclude unlikely tokens

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
    # Check for PEFT adapter files
    adapter_config = adapter_path / "adapter_config.json"
    adapter_model = adapter_path / "adapter_model.safetensors"
    adapter_model_bin = adapter_path / "adapter_model.bin"
    return adapter_config.exists() and (adapter_model.exists() or adapter_model_bin.exists())


def load_model():
    """Load the model with LoRA adapters."""
    print("Loading model...")

    # Check for CUDA availability
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Warning: CUDA not available. Running on CPU will be slow.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model in fp16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapters if available
    has_adapters = check_adapters()

    if has_adapters:
        print(f"  Base model: {MODEL}")
        print(f"  Adapters: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    else:
        print(f"  Model: {MODEL}")
        print("  Note: No adapters found. Using base model only.")
        print("  Run 'python train.py' to fine-tune the model.")

    model.eval()
    print("Model loaded!\n")
    return model, tokenizer


def format_prompt(messages: list[dict], tokenizer) -> str:
    """Format messages using the model's chat template."""
    try:
        # Use the tokenizer's built-in chat template
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

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def stream_response(model, tokenizer, messages: list[dict]):
    """Stream response tokens as they're generated."""
    prompt = format_prompt(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Create streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Generation kwargs
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print()  # Newline before response
    full_response = ""

    for text in streamer:
        print(text, end="", flush=True)
        full_response += text

    thread.join()
    print("\n")  # Newlines after response
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
    print("  Roleplay Chat - Charlie Kirk")
    print("=" * 50)
    print()

    # Load model
    model, tokenizer = load_model()

    # Initialize conversation with system prompt
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_history = 5  # Keep last N exchanges (reduced to prevent context pollution)

    print("Type '/help' for commands, '/quit' to exit.")
    print("-" * 50)

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
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
                print(f"Max tokens: {MAX_TOKENS}")
                print(f"Temperature: {TEMPERATURE}")
                if torch.cuda.is_available():
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                continue

            # Add user message
            conversation.append({"role": "user", "content": user_input})

            # Generate and stream response
            print("\nCharlie:", end="")
            response = stream_response(model, tokenizer, conversation)

            # Add assistant response to history
            conversation.append({"role": "assistant", "content": response})

            # Trim conversation history (keep system + last N exchanges)
            if len(conversation) > 1 + (max_history * 2):
                # Keep system prompt + last max_history user/assistant pairs
                conversation = [conversation[0]] + conversation[-(max_history * 2):]

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()

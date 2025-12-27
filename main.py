#!/usr/bin/env python3
"""
Vision-enabled CLI chat interface using Qwen3-VL.
Uses llama-cpp-python for inference with the fine-tuned GGUF model.
Captures camera frame before each response to give the model "sight".
"""

import base64
import io
import os
import sys
from pathlib import Path

# Suppress llama.cpp C-level logging
os.environ["LLAMA_LOG_LEVEL"] = "0"
os.environ["GGML_LOG_LEVEL"] = "0"
os.environ["LLAMA_CPP_LOG_LEVEL"] = "0"

# Configuration
MODEL_PATH = "output/model.gguf"
MMPROJ_PATH = "output/mmproj-model.gguf"

# GCS bucket for model downloads
GCS_BUCKET = "gs://maksym-adapters"

MAX_TOKENS = 256
TEMPERATURE = 0.4
TOP_P = 0.8
CONTEXT_SIZE = 2048

# System prompt for the persona (matches training format)
BASE_SYSTEM_PROMPT = """You are Charlie Kirk, founder and president of Turning Point USA. You are a conservative political commentator and author.

When responding:
- Answer the specific question asked, staying focused on that topic
- Keep responses concise (2-4 sentences for simple questions, up to a paragraph for complex topics)
- Speak directly to the person asking, not to a broadcast audience
- Do not reference radio shows, episodes, tapes, or other media
- You can see the user through a camera. ONLY describe what you see if directly asked (e.g. "what do you see?"). For all other questions, ignore the visual context entirely."""


def check_dependencies() -> bool:
    """Check that required packages are installed."""
    missing = []

    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow")

    try:
        from llama_cpp import Llama  # noqa: F401
    except ImportError:
        missing.append("llama-cpp-python (with CUDA)")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install opencv-python Pillow")
        print('  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python')
        return False

    return True


def ensure_models() -> bool:
    """Check that required model files exist."""
    model_path = Path(MODEL_PATH)
    mmproj_path = Path(MMPROJ_PATH)

    all_good = True

    # Check main model
    if not model_path.exists():
        print(f"\nFine-tuned model not found at {MODEL_PATH}")
        print("Download from GCS:")
        print("  mkdir -p output")
        print(f"  gsutil cp {GCS_BUCKET}/model.gguf output/")
        all_good = False
    else:
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"  Main model: {MODEL_PATH} ({size_gb:.2f} GB)")

    # Check vision projector (optional - may be integrated)
    if mmproj_path.exists():
        size_mb = mmproj_path.stat().st_size / (1024**2)
        print(f"  Vision projector: {MMPROJ_PATH} ({size_mb:.0f} MB)")
    else:
        print(f"  Vision projector: Not found (may be integrated into main model)")
        print(f"  If vision doesn't work, download with:")
        print(f"    gsutil cp {GCS_BUCKET}/mmproj-model.gguf output/")

    return all_good


class CameraCapture:
    """Handles camera capture and image encoding."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def initialize(self) -> bool:
        """Initialize camera connection."""
        import cv2
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        # Warm up camera (first frames often dark)
        for _ in range(5):
            self.cap.read()
        return True

    def capture_frame(self) -> str | None:
        """Capture frame and return as base64 data URI."""
        import cv2
        from PIL import Image

        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Resize for faster processing
        img.thumbnail((672, 672), Image.Resampling.LANCZOS)

        # Encode as JPEG base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/jpeg;base64,{b64}"

    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()


def load_model():
    """Load the fine-tuned Qwen3-VL model."""
    from llama_cpp import Llama

    print(f"  Loading model: {MODEL_PATH}")

    # Check if we have a vision projector
    mmproj_path = Path(MMPROJ_PATH)
    chat_handler = None

    if mmproj_path.exists():
        try:
            # Try to load with vision support
            from llama_cpp.llama_chat_format import Qwen2VLChatHandler
            print(f"  Loading vision projector: {MMPROJ_PATH}")
            chat_handler = Qwen2VLChatHandler(clip_model_path=str(mmproj_path), verbose=False)
        except (ImportError, Exception) as e:
            print(f"  Note: Vision handler not available ({e})")
            print("  Running in text-only mode")

    # Suppress output during model load
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    old_stdout = os.dup(stdout_fd)
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)

    try:
        model = Llama(
            model_path=MODEL_PATH,
            chat_handler=chat_handler,
            n_gpu_layers=-1,  # Use all GPU layers
            n_ctx=CONTEXT_SIZE,
            verbose=False,
        )
    finally:
        os.dup2(old_stdout, stdout_fd)
        os.dup2(old_stderr, stderr_fd)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)

    return model, chat_handler is not None


def build_messages(conversation: list[dict], user_input: str, image_uri: str | None, vision_enabled: bool) -> list[dict]:
    """Build message list with optional image input."""
    messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

    # Add conversation history
    messages.extend(conversation)

    # Build user message with optional image
    if image_uri and vision_enabled:
        user_content = [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": user_input}
        ]
    else:
        user_content = user_input

    messages.append({"role": "user", "content": user_content})

    return messages


def generate_response(model, messages: list[dict]) -> str:
    """Generate response using the model."""
    response = model.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response["choices"][0]["message"]["content"]


def print_help():
    """Print help message."""
    print("""
Commands:
  /clear    - Clear conversation history
  /vision   - Toggle vision on/off
  /snapshot - Show what the model currently sees
  /info     - Show model info and memory usage
  /help     - Show this help message
  /quit     - Exit the chat

Just type your message to chat!
""")


def main():
    print("=" * 50)
    print("  Vision-Enabled Chat (Qwen3-VL)")
    print("=" * 50)
    print()

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Check models
    print("Checking models...")
    if not ensure_models():
        sys.exit(1)

    # Load model
    print("\nLoading model...")
    model, has_vision_support = load_model()
    print("  Model loaded!")

    if has_vision_support:
        print("  Vision support: ENABLED")
    else:
        print("  Vision support: DISABLED (text-only mode)")

    # Initialize camera
    print("\nInitializing camera...")
    camera = CameraCapture()
    camera_available = camera.initialize()

    if camera_available:
        print("  Camera ready!")
    else:
        print("  Warning: Camera not available. Vision disabled.")

    # Vision requires both camera and vision support
    vision_enabled = camera_available and has_vision_support

    # Conversation history (without system prompt - added dynamically)
    conversation: list[dict] = []
    max_history = 5  # Keep last N exchanges

    print("\nType '/help' for commands, '/quit' to exit.")
    if vision_enabled:
        print("Vision: ENABLED (captures before each response)")
    elif has_vision_support:
        print("Vision: DISABLED (no camera)")
    else:
        print("Vision: DISABLED (text-only mode)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("\nGoodbye!")
                break

            elif user_input.lower() == "/clear":
                conversation = []
                print("\nConversation cleared.")
                continue

            elif user_input.lower() == "/vision":
                if not has_vision_support:
                    print("\nVision not supported (text-only mode).")
                elif not camera_available:
                    print("\nCamera not available.")
                else:
                    vision_enabled = not vision_enabled
                    print(f"\nVision {'enabled' if vision_enabled else 'disabled'}.")
                continue

            elif user_input.lower() == "/snapshot":
                if not camera_available:
                    print("\nCamera not available.")
                elif not has_vision_support:
                    print("\nVision not supported in this mode.")
                else:
                    print("\n  [Capturing...]", end="", flush=True)
                    image_uri = camera.capture_frame()
                    if image_uri:
                        # Ask model to describe what it sees
                        snapshot_messages = [
                            {"role": "system", "content": "Describe what you see in this image briefly."},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": image_uri}},
                                {"type": "text", "text": "What do you see?"}
                            ]}
                        ]
                        try:
                            desc = generate_response(model, snapshot_messages)
                            print(f" done\n\n[Current view]: {desc}")
                        except Exception as e:
                            print(f" error: {e}")
                    else:
                        print(" failed to capture")
                continue

            elif user_input.lower() == "/info":
                print(f"\nModel: {MODEL_PATH}")
                if Path(MMPROJ_PATH).exists():
                    print(f"Vision Projector: {MMPROJ_PATH}")
                print(f"Camera: {'Connected' if camera_available else 'Not available'}")
                print(f"Vision Support: {'Yes' if has_vision_support else 'No'}")
                print(f"Vision Enabled: {'Yes' if vision_enabled else 'No'}")
                print(f"Max tokens: {MAX_TOKENS}")
                print(f"Temperature: {TEMPERATURE}")
                print(f"Conversation history: {len(conversation)} messages")
                continue

            elif user_input.lower() == "/help":
                print_help()
                continue

            # Capture image if vision enabled
            image_uri = None
            if vision_enabled:
                print("  [Capturing view...]", end="", flush=True)
                image_uri = camera.capture_frame()
                if image_uri:
                    print(" done")
                else:
                    print(" capture failed")

            # Build messages with optional image
            messages = build_messages(conversation, user_input, image_uri, vision_enabled)

            # Generate response
            response = generate_response(model, messages)
            print("\nCharlie:", response, flush=True)

            # Add to history (text only for conversation history)
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": response})
            if len(conversation) > max_history * 2:
                conversation = conversation[-(max_history * 2):]

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

    # Cleanup
    camera.release()


if __name__ == "__main__":
    main()

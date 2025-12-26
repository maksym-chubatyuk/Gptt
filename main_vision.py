#!/usr/bin/env python3
"""
Vision-enabled CLI chat interface.
Uses llama-cpp-python for both:
- Fine-tuned Mistral GGUF (text generation)
- LLaVA GGUF (vision/image understanding)

Captures camera frame before each response to give the model "sight".
"""

import base64
import io
import sys
from pathlib import Path

import cv2
from PIL import Image

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava16ChatHandler
except ImportError:
    print("Error: llama-cpp-python not installed.")
    print("Install with CUDA support:")
    print('  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python')
    sys.exit(1)

# Configuration
TEXT_MODEL_PATH = "output/model.gguf"
VISION_MODEL_PATH = "models/llava-v1.6-mistral-7b.Q4_K_M.gguf"
VISION_CLIP_PATH = "models/mmproj-model-f16.gguf"

MAX_TOKENS = 256
TEMPERATURE = 0.4
TOP_P = 0.8
CONTEXT_SIZE = 2048

# System prompt for the persona (matches training)
BASE_SYSTEM_PROMPT = """You are Charlie Kirk, founder and president of Turning Point USA. You are a conservative political commentator and author.

You can see through a camera. When visual context is provided:
- Speak naturally as if you're directly observing
- Don't say "the description says" - you ARE seeing it
- If the question isn't visual, you may ignore the visual context

When responding:
- Answer the specific question asked, staying focused on that topic
- Keep responses concise (2-4 sentences for simple questions, up to a paragraph for complex topics)
- Speak directly to the person asking, not to a broadcast audience
- Use "I think" and "I believe" rather than rhetorical questions
- Do not reference radio shows, episodes, tapes, or other media"""


class CameraCapture:
    """Handles camera capture and image encoding."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def initialize(self) -> bool:
        """Initialize camera connection."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        # Warm up camera (first frames often dark)
        for _ in range(5):
            self.cap.read()
        return True

    def capture_frame(self) -> str | None:
        """Capture frame and return as base64 data URI."""
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Resize for faster processing (LLaVA handles up to 336x336 well)
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


def check_models() -> bool:
    """Check that all model files exist."""
    text_model = Path(TEXT_MODEL_PATH)
    vision_model = Path(VISION_MODEL_PATH)
    vision_clip = Path(VISION_CLIP_PATH)

    if not text_model.exists():
        print(f"Error: Text model not found at {TEXT_MODEL_PATH}")
        print("Run merge_and_convert.py first to create the GGUF model.")
        return False

    if not vision_model.exists():
        print(f"Error: Vision model not found at {VISION_MODEL_PATH}")
        print("Download LLaVA GGUF model:")
        print("  mkdir -p models")
        print("  wget -P models https://huggingface.co/cjpais/llava-v1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf")
        return False

    if not vision_clip.exists():
        print(f"Error: Vision CLIP model not found at {VISION_CLIP_PATH}")
        print("Download mmproj model:")
        print("  wget -P models https://huggingface.co/cjpais/llava-v1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf")
        return False

    return True


def load_text_model() -> Llama:
    """Load the fine-tuned text model (GGUF)."""
    print(f"  Loading text model: {TEXT_MODEL_PATH}")
    return Llama(
        model_path=TEXT_MODEL_PATH,
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=CONTEXT_SIZE,
        verbose=False,
    )


def load_vision_model() -> Llama:
    """Load LLaVA vision model (GGUF) with CLIP handler."""
    print(f"  Loading vision model: {VISION_MODEL_PATH}")
    print(f"  Loading CLIP model: {VISION_CLIP_PATH}")

    chat_handler = Llava16ChatHandler(clip_model_path=VISION_CLIP_PATH, verbose=False)

    return Llama(
        model_path=VISION_MODEL_PATH,
        chat_handler=chat_handler,
        n_gpu_layers=-1,
        n_ctx=CONTEXT_SIZE,
        verbose=False,
    )


def get_vision_description(vision_model: Llama, image_data_uri: str) -> str:
    """Get scene description using LLaVA GGUF."""
    response = vision_model.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": "Describe what you see in 2-3 sentences. Focus on people, their appearance, setting, and any notable objects."}
            ]
        }],
        max_tokens=150,
        temperature=0.1,  # Low temp for consistent descriptions
    )
    return response["choices"][0]["message"]["content"]


def build_messages(vision_desc: str | None, conversation: list[dict]) -> list[dict]:
    """Build message list with vision context injected into system prompt."""
    if vision_desc:
        system_content = f"{BASE_SYSTEM_PROMPT}\n\n[CURRENT VISUAL CONTEXT]\n{vision_desc}"
    else:
        system_content = BASE_SYSTEM_PROMPT

    return [{"role": "system", "content": system_content}] + conversation


def generate_response(text_model: Llama, messages: list[dict]) -> str:
    """Generate response using the fine-tuned text model."""
    response = text_model.create_chat_completion(
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
    print("  Vision-Enabled Chat - Charlie Kirk")
    print("=" * 50)
    print()

    # Check models exist
    if not check_models():
        sys.exit(1)

    # Load models
    print("Loading models...")
    text_model = load_text_model()
    vision_model = load_vision_model()
    print("  Models loaded!")

    # Initialize camera
    print("\nInitializing camera...")
    camera = CameraCapture()
    camera_available = camera.initialize()

    if camera_available:
        print("  Camera ready!")
    else:
        print("  Warning: Camera not available. Vision disabled.")

    vision_enabled = camera_available

    # Conversation history (without system prompt - added dynamically)
    conversation: list[dict] = []
    max_history = 5  # Keep last N exchanges

    print("\nType '/help' for commands, '/quit' to exit.")
    if vision_enabled:
        print("Vision: ENABLED (captures before each response)")
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
                if camera_available:
                    vision_enabled = not vision_enabled
                    print(f"\nVision {'enabled' if vision_enabled else 'disabled'}.")
                else:
                    print("\nCamera not available.")
                continue

            elif user_input.lower() == "/snapshot":
                if camera_available:
                    print("\n  [Capturing...]", end="", flush=True)
                    image_uri = camera.capture_frame()
                    if image_uri:
                        desc = get_vision_description(vision_model, image_uri)
                        print(f" done\n\n[Current view]: {desc}")
                    else:
                        print(" failed to capture")
                else:
                    print("\nCamera not available.")
                continue

            elif user_input.lower() == "/info":
                print(f"\nText Model: {TEXT_MODEL_PATH}")
                print(f"Vision Model: {VISION_MODEL_PATH}")
                print(f"Camera: {'Connected' if camera_available else 'Not available'}")
                print(f"Vision: {'Enabled' if vision_enabled else 'Disabled'}")
                print(f"Max tokens: {MAX_TOKENS}")
                print(f"Temperature: {TEMPERATURE}")
                print(f"Conversation history: {len(conversation)} messages")
                continue

            elif user_input.lower() == "/help":
                print_help()
                continue

            # Capture and describe scene BEFORE response
            vision_desc = None
            if vision_enabled and camera_available:
                print("  [Analyzing view...]", end="", flush=True)
                image_uri = camera.capture_frame()
                if image_uri:
                    vision_desc = get_vision_description(vision_model, image_uri)
                    print(" done")
                else:
                    print(" capture failed")

            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})

            # Build full message list with vision context
            messages = build_messages(vision_desc, conversation)

            # Generate response
            print("\nCharlie:", end=" ")
            response = generate_response(text_model, messages)
            print(response)

            # Add assistant response to history
            conversation.append({"role": "assistant", "content": response})

            # Trim conversation history (keep last max_history exchanges)
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

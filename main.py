#!/usr/bin/env python3
"""
Vision-enabled CLI chat interface.
Uses llama-cpp-python for both:
- Fine-tuned Mistral GGUF (text generation)
- LLaVA GGUF (vision/image understanding)

Captures camera frame before each response to give the model "sight".
Auto-downloads required models if not present.
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
TEXT_MODEL_PATH = "output/model.gguf"
VISION_MODEL_PATH = "models/ggml-model-q4_k.gguf"
VISION_CLIP_PATH = "models/mmproj-model-f16.gguf"

# Hugging Face repo for LLaVA vision model (public)
LLAVA_REPO = "mys/ggml_llava-v1.5-7b"
LLAVA_MODEL_FILE = "ggml-model-q4_k.gguf"
LLAVA_CLIP_FILE = "mmproj-model-f16.gguf"

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


def download_hf_file(repo_id: str, filename: str, dest: Path, description: str, min_size_mb: int = 100) -> bool:
    """Download a file from Hugging Face using huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  Error: huggingface_hub not installed")
        print("  Run: pip install huggingface_hub")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Remove failed previous download
    if dest.exists() and dest.stat().st_size < min_size_mb * 1024 * 1024:
        dest.unlink()

    print(f"  Downloading {description}...")
    print(f"    Repo: {repo_id}")
    print(f"    File: {filename}")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest.parent,
            local_dir_use_symlinks=False,
        )

        # Move to expected location if needed
        downloaded = Path(downloaded_path)
        if downloaded != dest:
            import shutil
            shutil.move(str(downloaded), str(dest))

    except Exception as e:
        print(f"  Error downloading: {e}")
        print("  You may need to login: huggingface-cli login")
        return False

    # Verify download succeeded
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        if size_mb < min_size_mb:
            print(f"  Error: Download incomplete (got {size_mb:.1f} MB, expected >{min_size_mb} MB)")
            dest.unlink()
            return False
        print(f"  Done: {size_mb:.0f} MB")
        return True

    return False


def ensure_models() -> bool:
    """Download any missing models."""
    text_model = Path(TEXT_MODEL_PATH)
    vision_model = Path(VISION_MODEL_PATH)
    vision_clip = Path(VISION_CLIP_PATH)

    all_good = True

    # Check fine-tuned text model (user downloads manually)
    if not text_model.exists():
        print(f"\nFine-tuned text model not found at {TEXT_MODEL_PATH}")
        print("Download from GCS:")
        print("  mkdir -p output")
        print("  gsutil cp gs://maksym-adapters/model.gguf output/")
        all_good = False
    else:
        size_gb = text_model.stat().st_size / (1024**3)
        print(f"  Text model: {TEXT_MODEL_PATH} ({size_gb:.2f} GB)")

    # Check/download LLaVA vision model (~4GB)
    if not vision_model.exists() or vision_model.stat().st_size < 1000 * 1024 * 1024:
        print("\nLLaVA vision model not found.")
        if not download_hf_file(LLAVA_REPO, LLAVA_MODEL_FILE, vision_model, "LLaVA model (~4GB)", min_size_mb=3500):
            all_good = False
    else:
        size_gb = vision_model.stat().st_size / (1024**3)
        print(f"  Vision model: {VISION_MODEL_PATH} ({size_gb:.2f} GB)")

    # Check/download LLaVA CLIP model (~600MB)
    if not vision_clip.exists() or vision_clip.stat().st_size < 100 * 1024 * 1024:
        print("\nLLaVA CLIP model not found.")
        if not download_hf_file(LLAVA_REPO, LLAVA_CLIP_FILE, vision_clip, "CLIP model (~600MB)", min_size_mb=500):
            all_good = False
    else:
        size_mb = vision_clip.stat().st_size / (1024**2)
        print(f"  CLIP model: {VISION_CLIP_PATH} ({size_mb:.0f} MB)")

    return all_good


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
        from llama_cpp.llama_chat_format import Llava15ChatHandler  # noqa: F401
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


def load_text_model():
    """Load the fine-tuned text model (GGUF)."""
    from llama_cpp import Llama

    print(f"  Loading text model: {TEXT_MODEL_PATH}")
    return Llama(
        model_path=TEXT_MODEL_PATH,
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=CONTEXT_SIZE,
        verbose=False,
    )


def load_vision_model():
    """Load LLaVA vision model (GGUF) with CLIP handler."""
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler

    print(f"  Loading vision model: {VISION_MODEL_PATH}")
    print(f"  Loading CLIP model: {VISION_CLIP_PATH}")

    # Suppress all output during handler creation
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    old_stdout = os.dup(stdout_fd)
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)

    try:
        chat_handler = Llava15ChatHandler(clip_model_path=VISION_CLIP_PATH, verbose=False)
        model = Llama(
            model_path=VISION_MODEL_PATH,
            chat_handler=chat_handler,
            n_gpu_layers=-1,
            n_ctx=CONTEXT_SIZE,
            verbose=False,
        )
    finally:
        os.dup2(old_stdout, stdout_fd)
        os.dup2(old_stderr, stderr_fd)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)

    return model


class VisionServer:
    """Runs LLaVA in a subprocess to isolate debug output."""

    def __init__(self, model_path: str, clip_path: str):
        self.model_path = model_path
        self.clip_path = clip_path
        self.process = None

    def start(self):
        """Start the vision server subprocess."""
        import subprocess

        # Python script that runs as server - uses stdin/stdout for IPC
        server_code = '''
import sys, os

# Redirect stderr to devnull to suppress debug
sys.stderr = open(os.devnull, "w")

# Save stdout for IPC, then redirect to devnull during model load
real_stdout = os.fdopen(os.dup(sys.stdout.fileno()), "w")
sys.stdout = open(os.devnull, "w")

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

model_path = sys.argv[1]
clip_path = sys.argv[2]

handler = Llava15ChatHandler(clip_model_path=clip_path, verbose=False)
model = Llama(model_path=model_path, chat_handler=handler, n_gpu_layers=-1, n_ctx=2048, verbose=False)

# Signal ready
real_stdout.write("READY\\n")
real_stdout.flush()

# Process requests from stdin
for line in sys.stdin:
    image_uri = line.strip()
    if image_uri == "QUIT":
        break

    try:
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": "In 10 words or less, what is in this image?"}
            ]}],
            max_tokens=30,
            temperature=0.1,
        )
        result = response["choices"][0]["message"]["content"].replace("\\n", " ")
    except Exception as e:
        result = f"Vision error: {e}"

    real_stdout.write(result + "\\n")
    real_stdout.flush()
'''

        self.process = subprocess.Popen(
            [sys.executable, "-c", server_code, self.model_path, self.clip_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        # Wait for ready signal
        ready = self.process.stdout.readline()
        if "READY" not in ready:
            raise RuntimeError(f"Vision server failed to start: {ready}")

    def describe(self, image_data_uri: str) -> str:
        """Get vision description."""
        if not self.process:
            return "Vision server not running"

        self.process.stdin.write(image_data_uri + "\n")
        self.process.stdin.flush()
        return self.process.stdout.readline().strip()

    def stop(self):
        """Stop the vision server."""
        if self.process:
            try:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except:
                self.process.kill()


_vision_server = None

def get_vision_description(_vision_model, image_data_uri: str) -> str:
    """Get scene description using LLaVA via subprocess."""
    global _vision_server
    if _vision_server:
        return _vision_server.describe(image_data_uri)
    return "Vision not available"




def build_messages(vision_desc: str | None, conversation: list[dict]) -> list[dict]:
    """Build message list with vision context injected into system prompt."""
    if vision_desc:
        system_content = f"{BASE_SYSTEM_PROMPT}\n\n[CURRENT VISUAL CONTEXT]\n{vision_desc}"
    else:
        system_content = BASE_SYSTEM_PROMPT

    return [{"role": "system", "content": system_content}] + conversation


def generate_response(text_model, messages: list[dict]) -> str:
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
    print("  Vision-Enabled Chat")
    print("=" * 50)
    print()

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Check/download models
    print("Checking models...")
    if not ensure_models():
        sys.exit(1)

    # Load models
    print("\nLoading models...")
    text_model = load_text_model()
    print("  Starting vision server...")
    global _vision_server
    _vision_server = VisionServer(VISION_MODEL_PATH, VISION_CLIP_PATH)
    _vision_server.start()
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
                        desc = get_vision_description(None, image_uri)
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
                    vision_desc = get_vision_description(None, image_uri)
                    print(" done")
                else:
                    print(" capture failed")

            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})

            # Build full message list with vision context
            messages = build_messages(vision_desc, conversation)

            # Generate response
            response = generate_response(text_model, messages)
            print("\nCharlie:", response, flush=True)

            # Add to history
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
    if _vision_server:
        _vision_server.stop()


if __name__ == "__main__":
    main()

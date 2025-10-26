"""Configuration constants and environment variables."""
import os
import logging

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "memories.sqlite")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Audio configuration
SAMPLE_RATE = 16000
AUDIO_FILE = "temp_audio.wav"
RECORD_DURATION = 5  # Seconds

# User configuration
DEFAULT_SPEAKER = "test1"

# Whisper model configuration
WHISPER_MODEL_PATH = (
    "/mnt/ssd/huggingface/hub/"
    "models--openai--whisper-large-v3-turbo/"
    "snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
)

# Conversation memory configuration
DEFAULT_HISTORY_WINDOW = 5  # short-term window size

# MemoBase configuration
MEMOBASE_PROJECT_URL = os.getenv("MEMOBASE_PROJECT_URL", "http://localhost:8019")
MEMOBASE_API_KEY = os.getenv("MEMOBASE_API_KEY", "secret")
MEMOBASE_API_VERSION = os.getenv("MEMOBASE_API_VERSION", "api/v1")
MEMOBASE_TIMEOUT = int(os.getenv("MEMOBASE_TIMEOUT", "30"))
MEMOBASE_BASE_URL = f"{MEMOBASE_PROJECT_URL.rstrip('/')}/{MEMOBASE_API_VERSION.strip('/')}"

DEFAULT_MEMORY_PROMPT = "Only refer to the memory if it's relevant to user's input."
DEFAULT_MAX_CONTEXT_SIZE = 1000

# Memory cache configuration
MEMORY_CACHE_FILE = os.getenv("MEMORY_CACHE_FILE", os.path.join(BASE_DIR, "memory_cache.json"))

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma3:1b"
OLLAMA_STREAM = True
OLLAMA_TEMPERATURE = 0.7
OLLAMA_MAX_TOKENS = 256

# Personality model paths
PERSONALITY_MODELS_DIR = os.path.join(BASE_DIR, "data", "models")

# Logging configuration
# Set to INFO to see audio debugging information, ERROR to suppress
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

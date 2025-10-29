# AI Voice Chatbot with Personality Analysis

An intelligent voice-powered chatbot that combines speech recognition, personality analysis, and long-term memory to provide personalized conversational experiences.

## Features

- **Voice Interaction**: Real-time audio recording and speech-to-text using OpenAI Whisper
- **Personality Analysis**: Big Five personality trait detection using BERT-based models
- **Long-term Memory**: MemoBase integration for persistent conversation context
- **Text-to-Speech**: Natural voice responses using pyttsx3
- **Intelligent Context**: Short-term and long-term memory management
- **Performance Monitoring**: Built-in timing metrics for all operations
- **Robust Audio**: Timeout protection and automatic retry mechanisms

## Architecture

```
┌─────────────────┐
│  Audio Input    │
│  (Microphone)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Speech-to-Text │─────▶│  Whisper Model   │
│  (Whisper)      │      │  (Large-v3-Turbo)│
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Personality    │─────▶│  BERT Model      │
│  Analysis       │      │  (Big Five)      │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Context        │─────▶│  MemoBase API    │
│  Injecting      │      │  (qwen2.5:7b)    │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  LLM Chat       │─────▶│  Ollama          │
│  (Response Gen) │      │  (Gemma 3:1b)    │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐
│  Text-to-Speech │
│  (pyttsx3)      │
└─────────────────┘
```

## Project Structure

```
chatbot/
├── main.py                   # Main application entry point
├── sync_memory_cache.py      # Utility to sync cache to MemoBase
├── memories.sqlite           # SQLite database for user data
├── memory_cache.json         # Conversation cache
├── modules/
│   ├── audio.py             # Audio recording and TTS
│   ├── config.py            # Configuration constants
│   ├── database.py          # SQLAlchemy models and DB ops
│   ├── llm.py               # Ollama LLM integration
│   ├── memory.py            # MemoBase memory management
│   ├── personality.py       # Big Five personality analysis
│   ├── speech.py            # Whisper speech-to-text
│   └── timing.py            # Performance monitoring
├── legacy/                   # Legacy/backup files
│   ├── app_new_legacy.py    # Old monolithic version
│   └── personality_old.py   # Old personality module
└── data/
    └── models/              # Personality model cache
```
## TODOs
 - update personality analysis part
 - multi-agent part
 - RAG part for better semantic context (optional)
 - update long term memory terms
 - more tests

## Installation

### Prerequisites

- Python 3.10
- CUDA-capable GPU (optional, for faster inference)
- Microphone for audio input
- Ollama running locally (port 11434)
- MemoBase instance 

### Dependencies

```bash
pip install torch transformers
pip install sounddevice scipy numpy
pip install pyttsx3
pip install sqlalchemy
pip install pandas
pip install openai requests
```

### Model Setup

1. **Whisper Model**: Download Whisper Large-v3-Turbo
```bash
# Update WHISPER_MODEL_PATH in modules/config.py
# Default: /mnt/ssd/huggingface/hub/models--openai--whisper-large-v3-turbo/...
```

2. **Personality Model**: Automatically downloads on first run
```python
# Model: Minej/bert-base-personality
# Cached in: data/models/
```

3. **Ollama LLM**: Start Ollama server
```bash
ollama serve
ollama pull gemma3:1b
```

## Configuration

### Environment Variables

### Audio Configuration

Edit [modules/config.py](modules/config.py):

```python
# Audio settings
SAMPLE_RATE = 16000        # Sample rate in Hz
RECORD_DURATION = 5        # Recording duration in seconds
AUDIO_FILE = "temp_audio.wav"

# User settings
DEFAULT_SPEAKER = "test1"  # Default speaker name

# History
DEFAULT_HISTORY_WINDOW = 5 # Conversation rounds to keep
```

### LLM Configuration

```python
# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma3:1b"
OLLAMA_STREAM = True
OLLAMA_TEMPERATURE = 0.7
OLLAMA_MAX_TOKENS = 256
```

## Usage

### Basic Usage

```bash
python main.py
```

### With Custom History Window

```bash
python main.py --history-window 10
```

### Debug Mode

```bash
python main.py --debug
```

### Sync Memory Cache to MemoBase

```bash
python sync_memory_cache.py --batch-size 10
```

### Interactive Commands

- Type `r` to start recording (5 seconds)
- Type `q` to quit the application
- Select audio input device from the list

### Example Session

```
Type 'r' to record, 'q' to quit.
r
Available input devices:
0: NVIDIA Jetson AGX Orin APE: - (hw:1,0) (channels: 16)
...
20: OD-WB01: USB Audio (hw:2,0) (channels: 2)
Enter input device index (or -1 for default): 20

Recording for 5 seconds...
Q: How are you today?
A: I'm doing great! How can I help you?

[Processing completed in 2.3456s]
```

## Features in Detail

### 1. Speech Recognition

- Uses OpenAI Whisper Large-v3-Turbo
- Supports GPU acceleration (CUDA)
- Automatic silence detection
- Robust error handling with timeouts

### 2. Personality Analysis

Analyzes user speech for Big Five personality traits:
- **Extraversion**: Social engagement level
- **Neuroticism**: Emotional stability
- **Agreeableness**: Cooperativeness
- **Conscientiousness**: Organization and responsibility
- **Openness**: Creativity and curiosity

Traits are stored per user and used to personalize responses.

### 3. Memory Management

**Short-term Memory**:
- Configurable conversation window (default: 5 rounds)
- Sliding window mechanism
- Included in every prompt context

**Long-term Memory** (via MemoBase):
- Persistent across sessions
- Semantic search for relevant context
- User-specific memory isolation
- Automatic cache management

### 4. Audio Robustness

Enhanced audio handling with:
- **Timeout protection**: 7-second max wait (5s recording + 2s margin)
- **Non-blocking stop**: Prevents hanging on sd.stop()
- **Automatic retry**: Up to 2 retries on failure
- **Device validation**: Checks device capabilities before recording
- **Detailed logging**: Device info, audio statistics


## Credits

- **OpenAI Whisper**: Speech recognition model
- **Minej/bert-base-personality**: Big Five personality model
- **Ollama**: Local LLM inference
- **MemoBase**: Long-term memory system

## License




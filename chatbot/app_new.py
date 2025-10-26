import argparse
import copy
import io
import json
import logging
import os
import pickle
import random
import sqlite3
import threading
import time as time_module
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Dict, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import sounddevice as sd
import scipy.io.wavfile as wavfile
import torch
from sqlalchemy import Column, Integer, Numeric, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from transformers import pipeline
from ollama import chat
import pyttsx3

from openai import OpenAI

# ----------------------- Configuration -----------------------
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# cv2.ocl.setUseOpenCL(False)  # Legacy face-recognition configuration (disabled)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "memories.sqlite")
DATABASE_URL = f"sqlite:///{DB_PATH}"
SAMPLE_RATE = 16000
AUDIO_FILE = "temp_audio.wav"
RECORD_DURATION = 5  # Seconds

DEFAULT_SPEAKER = "test1"

current_speaker = DEFAULT_SPEAKER
preferences: dict[str, Any] = {}
tts_engine = None  # Global TTS engine

WHISPER_MODEL_PATH = (
    "/mnt/ssd/huggingface/hub/"
    "models--openai--whisper-large-v3-turbo/"
    "snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
)


# Conversation memory (short- and long-term)
DEFAULT_HISTORY_WINDOW = 5  # short-term window size
history_window_size = DEFAULT_HISTORY_WINDOW


@dataclass
class ConversationState:
    history: List[Dict[str, str]] = field(default_factory=list)

conversation_states: Dict[str, ConversationState] = {}

def get_conversation_state(speaker_key: str) -> ConversationState:
    state = conversation_states.get(speaker_key)
    if state is None:
        state = ConversationState()
        conversation_states[speaker_key] = state
    return state

# ----------------------- Model Loading -----------------------
def load_pickle_model(path: str):
    try:
        abs_path = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        with open(abs_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model {path}: {e}")
        raise

cEXT = load_pickle_model("data/models/cEXT.p")
cNEU = load_pickle_model("data/models/cNEU.p")
cAGR = load_pickle_model("data/models/cAGR.p")
cCON = load_pickle_model("data/models/cCON.p")
cOPN = load_pickle_model("data/models/cOPN.p")
vectorizer_31 = load_pickle_model("data/models/vectorizer_31.p")
vectorizer_30 = load_pickle_model("data/models/vectorizer_30.p")
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------- Database Setup -----------------------
Base = declarative_base()

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, autoincrement=True)
    openness = Column(Numeric, nullable=True)
    conscientiousness = Column(Numeric, nullable=True)
    extraversion = Column(Numeric, nullable=True)
    agreeableness = Column(Numeric, nullable=True)
    neuroticism = Column(Numeric, nullable=True)
    name = Column(String, nullable=True)
    # nickname = Column(String, nullable=True)
    # image = Column(String, nullable=True)
    # preferences = Column(Text)

def init_db():
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                chat_session_id TEXT,
                text TEXT,
                embedding TEXT,
                entity TEXT,
                current_summary TEXT
            )
        """)
        conn.commit()
    return sessionmaker(bind=engine)

Session = init_db()
db_session = Session()
# ----------------------- Memobase Setup -----------------------

MEMOBASE_PROMPT = """

--# ADDITIONAL INFO #--
{user_context}
{additional_memory_prompt}
--# DONE #--"""

DEFAULT_MEMORY_PROMPT = "Only refer to the memory if it's relevant to user's input."
DEFAULT_MAX_CONTEXT_SIZE = 1000

MEMOBASE_PROJECT_URL = os.getenv("MEMOBASE_PROJECT_URL", "http://localhost:8019")
MEMOBASE_API_KEY = os.getenv("MEMOBASE_API_KEY", "secret")
MEMOBASE_API_VERSION = os.getenv("MEMOBASE_API_VERSION", "api/v1")
MEMOBASE_TIMEOUT = int(os.getenv("MEMOBASE_TIMEOUT", "30"))
MEMOBASE_BASE_URL = f"{MEMOBASE_PROJECT_URL.rstrip('/')}/{MEMOBASE_API_VERSION.strip('/')}"

MEMORY_CACHE_FILE = os.getenv("MEMORY_CACHE_FILE", os.path.join(BASE_DIR, "memory_cache.json"))


class MemoBaseAPIError(Exception):
    pass


memo_session = requests.Session()
memo_session.headers.update({"Authorization": f"Bearer {MEMOBASE_API_KEY}"})


def string_to_uuid(value: str, salt: str = "memobase_client") -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{value}{salt}"))


def _handle_memobase_response(response: requests.Response) -> Any:
    try:
        payload = response.json()
    except ValueError as exc:
        raise MemoBaseAPIError("Invalid JSON response from MemoBase API") from exc
    if payload.get("errno") != 0:
        raise MemoBaseAPIError(payload.get("errmsg", "MemoBase API error"))
    return payload.get("data")


def memobase_request(
    method: str,
    path: str,
    *,
    params: Optional[dict] = None,
    json_payload: Optional[dict] = None,
    allow_404: bool = False,
) -> Any:
    url = f"{MEMOBASE_BASE_URL}{path}"
    response = memo_session.request(
        method,
        url,
        params=params,
        json=json_payload,
        timeout=MEMOBASE_TIMEOUT,
    )
    if allow_404 and response.status_code == 404:
        return None
    response.raise_for_status()
    return _handle_memobase_response(response)


def ensure_memobase_user(user_uuid: str) -> None:
    if not user_uuid:
        return
    existing = memobase_request("GET", f"/users/{user_uuid}", allow_404=True)
    if existing is not None:
        return
    memobase_request("POST", "/users", json_payload={"id": user_uuid, "data": None})


def fetch_memobase_context(
    user_uuid: str,
    max_token_size: int = DEFAULT_MAX_CONTEXT_SIZE,
    chats: Optional[list] = None,
) -> str:
    params: dict[str, Any] = {"max_token_size": max(0, max_token_size)}
    if chats:
        params["chats_str"] = json.dumps(chats, ensure_ascii=False)
    data = memobase_request("GET", f"/users/context/{user_uuid}", params=params)
    if not data:
        return ""
    return data.get("context", "") or ""


def build_context_prompt(context: str, additional_prompt: str = DEFAULT_MEMORY_PROMPT) -> str:
    return MEMOBASE_PROMPT.format(
        user_context=context,
        additional_memory_prompt=additional_prompt,
    ).strip()


def inject_memobase_context(messages: List[Dict[str, str]], context: str) -> List[Dict[str, str]]:
    if not context:
        return messages
    prompt_text = build_context_prompt(context)
    if not messages:
        return [{"role": "system", "content": prompt_text}]
    first_message = messages[0]
    if first_message.get("role") == "system":
        first_message["content"] = (first_message.get("content") or "") + "\n" + prompt_text
    else:
        messages.insert(0, {"role": "system", "content": prompt_text})
    return messages


def prepare_recent_chats(messages: List[Dict[str, str]], max_items: int = 6) -> List[Dict[str, str]]:
    filtered = [
        {"role": msg.get("role", ""), "content": msg.get("content", "")}
        for msg in messages
        if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]
    return filtered[-max_items:]


def append_chat_to_cache(
    user_uuid: str,
    user_name: str,
    user_text: str,
    assistant_text: str,
    speech_duration: float,
    llm_duration: float,
) -> None:
    if not user_uuid or (not user_text and not assistant_text):
        return
    speech_duration = round(speech_duration, 2)
    llm_duration = round(llm_duration, 2)
    total_duration = round(speech_duration + llm_duration, 2)
    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "user_uuid": user_uuid,
        "user_name": user_name,
        "timings": {
            "speech_to_text": speech_duration,
            "llm_generation": llm_duration,
            "total": total_duration,
        },
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }
    cache_dir = os.path.dirname(MEMORY_CACHE_FILE)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    existing_entries: List[Dict[str, Any]] = []
    if os.path.exists(MEMORY_CACHE_FILE):
        try:
            with open(MEMORY_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                existing_entries = data
            else:
                logger.error("Unexpected data format in memory cache; resetting file.")
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read memory cache file; overwriting. %s", exc)
    existing_entries.append(entry)
    with open(MEMORY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_entries, f, ensure_ascii=False, indent=2)



# ----------------------- Memobase Setup -----------------------
stream = True
model = "gemma3:1b"

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def chat(
    messages: List[Dict[str, str]],
    close_session: bool = False,
    use_users: bool = True,
) -> Tuple[str, Optional[str]]:
    """使用 MemoBase 长期记忆增强的聊天接口。"""
    last_user_message = ""
    for item in reversed(messages):
        if item.get("role") == "user":
            last_user_message = item.get("content", "")
            break

    user_uuid: Optional[str] = None
    messages_for_llm = copy.deepcopy(messages)

    if use_users and current_speaker:
        user_uuid = string_to_uuid(current_speaker)
        try:
            ensure_memobase_user(user_uuid)
        except Exception as exc:
            logger.error(f"Failed to ensure MemoBase user {current_speaker}: {exc}")
            user_uuid = None
        if user_uuid:
            try:
                chats_for_context = prepare_recent_chats(messages)
                context_text = fetch_memobase_context(
                    user_uuid,
                    DEFAULT_MAX_CONTEXT_SIZE,
                    chats=chats_for_context,
                )
            except Exception as exc:
                logger.error(f"Failed to fetch MemoBase context for {current_speaker}: {exc}")
                context_text = ""
            messages_for_llm = inject_memobase_context(messages_for_llm, context_text)

    try:
        response = client.chat.completions.create(
            messages=messages_for_llm,
            model=model,
            stream=stream,
            user=current_speaker if use_users else None,
        )
    except Exception as exc:
        logger.error(f"MemoBase chat request failed: {exc}")
        return ""

    assistant_reply = ""
    if stream:
        collected_chunks: list[str] = []
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            delta_text = getattr(delta, "content", None)
            if not delta_text:
                continue
            collected_chunks.append(delta_text)
            print(delta_text, end="", flush=True)
        print()
        assistant_reply = "".join(collected_chunks).strip()
    else:
        if response.choices:
            assistant_reply = (response.choices[0].message.content or "").strip()
            if assistant_reply:
                print(assistant_reply)

    return assistant_reply, user_uuid



def format_short_term_memory(history: List[Dict[str, str]]) -> str:
    """Format stored conversation turns for inclusion in the system prompt."""
    if not history:
        return "None."
    formatted = []
    for entry in history:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        formatted.append(f"{label}: {content}")
    return "\n".join(formatted) if formatted else "None."


# ----------------------- TTS Engine Setup -----------------------
def init_tts_engine():
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 200)
        tts_engine.setProperty('volume', 0.8)
        voices = tts_engine.getProperty('voices')
        if len(voices) > 1:
            tts_engine.setProperty('voice', voices[0].id)
    except Exception as e:
        logger.error(f"Failed to initialize TTS engine: {e}")
        tts_engine = None

def cleanup_tts_engine():
    global tts_engine
    if tts_engine:
        try:
            tts_engine.stop()
            if getattr(tts_engine, "_inLoop", False):
                tts_engine.endLoop()
        except Exception as e:
            logger.error(f"Failed to clean up TTS engine: {e}")
        tts_engine = None

# ----------------------- Audio Device Selection -----------------------
def select_input_device():
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    if not input_devices:
        logger.error("No input audio devices found.")
        return None
    print("Available input devices:")
    for i, d in enumerate(input_devices):
        print(f"{i}: {d['name']} (channels: {d['max_input_channels']})")
    try:
        idx = int(input("Enter input device index (or -1 for default): "))
        if idx == -1:
            return None
        return input_devices[idx]['index']
    except (ValueError, IndexError):
        logger.error("Invalid device index. Using default.")
        return None

# ----------------------- Audio Recording -----------------------
def record_audio(device_index: Optional[int] = None):
    try:
        print(f"Recording for {RECORD_DURATION} seconds...")
        audio_data = sd.rec(
            int(RECORD_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            device=device_index
        )
        sd.wait()
        if audio_data is None or not np.any(audio_data):
            logger.error("No audio data recorded or data is all zeros.")
            return False
        wavfile.write(AUDIO_FILE, SAMPLE_RATE, audio_data)
        print(f"Audio recording saved to {AUDIO_FILE}")
        return True
    except Exception as e:
        logger.error(f"Audio recording failed: {e}")
        return False

def toggle_recording():
    device_index = select_input_device()
    if device_index is None and not any(d['max_input_channels'] > 0 for d in sd.query_devices()):
        logger.error("No valid input device. Aborting recording.")
        return False
    return record_audio(device_index)

def transcribe_whisper(audio_file: str, pipe):
    try:
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        if not audio_data:
            logger.error("Audio file is empty.")
            return ""
        audio_file_io = io.BytesIO(audio_data)
        audio_file_io.name = "audio.wav"
        #outputs = pipe(audio_data, chunk_length_s=10, batch_size=24, return_timestamps=False)
        outputs = pipe(audio_data, batch_size=48, return_timestamps=False)
        transcription = outputs["text"]
        return transcription
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return ""

def predict_personality(text: str) -> list:
    try:
        sentences = text.split(". ")
        text_vector_31 = vectorizer_31.transform(sentences)
        text_vector_30 = vectorizer_30.transform(sentences)
        ext = cEXT.predict(text_vector_31)
        neu = cNEU.predict(text_vector_30)
        agr = cAGR.predict(text_vector_31)
        con = cCON.predict(text_vector_31)
        opn = cOPN.predict(text_vector_31)
        return [ext[0], neu[0], agr[0], con[0], opn[0]]
    except Exception as e:
        logger.error(f"Personality prediction failed: {e}")
        return [0, 0, 0, 0, 0]

def store_personality_traits(predictions: list) -> None:
    """Persist Big Five personality traits for the current speaker."""
    global current_speaker
    if not current_speaker:
        logger.warning("No current speaker identified. Skipping Big Five persistence.")
        return

    if len(predictions) != 5:
        logger.error("Unexpected personality predictions shape; expected 5 values.")
        return

    extraversion, neuroticism, agreeableness, conscientiousness, openness = predictions
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT id FROM user WHERE name = ?", (current_speaker,))
            row = c.fetchone()
            if row:
                c.execute(
                    """
                    UPDATE user
                    SET openness = ?, conscientiousness = ?, extraversion = ?, agreeableness = ?, neuroticism = ?
                    WHERE name = ?
                    """,
                    (openness, conscientiousness, extraversion, agreeableness, neuroticism, current_speaker),
                )
            else:
                c.execute(
                    """
                    INSERT INTO user (name, openness, conscientiousness, extraversion, agreeableness, neuroticism)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (current_speaker, openness, conscientiousness, extraversion, agreeableness, neuroticism),
                )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to persist Big Five traits: {e}")

def fetch_user_data():
    if current_speaker:
        try:
            user = db_session.query(User).filter_by(name=current_speaker).first()
            if user and user.preferences:
                return json.loads(user.preferences)
        except Exception as e:
            logger.error(f"Failed to fetch user data: {e}")
    return {}

def process_audio(audio_file: str):
    global current_speaker, preferences, tts_engine, history_window_size
    if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
        logger.error(f"Audio file {audio_file} is missing or empty.")
        return

    use_gpu = torch.cuda.is_available()
    pipeline_device = 0 if use_gpu else "cpu"
    torch_dtype = torch.float16 if use_gpu else torch.float32
    if not os.path.isdir(WHISPER_MODEL_PATH):
        raise FileNotFoundError(f"Whisper model directory not found: {WHISPER_MODEL_PATH}")
    try:
        pipeline_kwargs: Dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": WHISPER_MODEL_PATH,
            "dtype": torch_dtype,
            "device": pipeline_device,
            "model_kwargs": {"attn_implementation": "sdpa"},
        }
        pipe = pipeline(**pipeline_kwargs)
    except Exception as e:
        logger.error(f"Failed to load Whisper pipeline: {e}")
        return

    try:
        if not tts_engine:
            logger.error("TTS engine not initialized.")
            init_tts_engine()
        if not tts_engine:
            logger.error("TTS engine unavailable, skipping speech.")
            return

        GREETINGS = [
            "Give me a sec to think about that.",
            "Let me process that real quick.",
            #"That's a good one! Thinking...",
            "Just a moment, I'm working on it.",
            "Let me figure this out for you.",
            "Hold on, I'll get right back to you.",
            "One moment while I put this together.",
        ]

        greeting = f"Hello {current_speaker}. {random.choice(GREETINGS)}" if current_speaker else f"Sorry, I couldn't recognize you. {random.choice(GREETINGS)}"
        try:
            tts_engine.say(greeting)
            tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Failed to speak greeting: {e}")

        speech_start = time_module.perf_counter()
        transcription = transcribe_whisper(audio_file, pipe)
        speech_duration = time_module.perf_counter() - speech_start
        text = transcription.strip()
        if not text:
            logger.error("Transcription is empty.")
            return
        print("Q: ",text)
        predictions = predict_personality(text)
        df = pd.DataFrame({"r": predictions, "theta": ["EXT", "NEU", "AGR", "CON", "OPN"]})

        speaker_key = current_speaker or "anonymous"
        state = get_conversation_state(speaker_key)
        history = state.history
        recent_history = history[-2 * history_window_size :] if history_window_size > 0 else history

        user_prompt_text = text
        personality_context = df.to_string()
        preferences_context = json.dumps(preferences) if preferences else "None."

        memory_context = format_short_term_memory(recent_history)
        system_prompt = "\n".join(
            [
                "You are Hackcelerate, a helpful healthcare assistant.",
                "Leverage the short-term memory when it helps the current request.",
                "Short-term memory:",
                memory_context,
                "Big Five Personality traits inferred for the user, use them when necessary:",
                personality_context,
                "Known user preferences:",
                preferences_context,
            ]
        )

        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_text},
        ]

        llm_start = time_module.perf_counter()
        response_content, user_uuid = chat(
            chat_messages,
            close_session=False,
            use_users=bool(current_speaker),
        )
        llm_duration = time_module.perf_counter() - llm_start
        if not response_content:
            logger.error("LLM response empty.")
            return

        store_personality_traits(predictions)

        if user_uuid and response_content:
            append_chat_to_cache(
                user_uuid,
                current_speaker or DEFAULT_SPEAKER,
                user_prompt_text,
                response_content,
                speech_duration,
                llm_duration,
            )

        history.append({"role": "user", "content": user_prompt_text})
        history.append({"role": "assistant", "content": response_content})
        if history_window_size > 0:
            max_items = 2 * history_window_size
            if len(history) > max_items:
                del history[:-max_items]

        try:
            tts_engine.say(response_content)
            tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Failed to speak LLM response: {e}")

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
    finally:
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if os.path.exists(audio_file):
            os.remove(audio_file)


# ----------------------- Main Program -----------------------
def main():
    parser = argparse.ArgumentParser(description="Audio processing with optional face recognition")
    parser.add_argument("--use-face-recognition", action="store_true", help="Enable face recognition")
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help="Number of recent conversation rounds to retain for short-term memory (0 disables).",
    )
    args = parser.parse_args()

    global history_window_size
    history_window_size = max(0, args.history_window)

    init_tts_engine()

    print("Type 'r' to record, 'q' to quit.")
    while True:
        try:
            command = input().strip().lower()
            if command == 'q':
                cleanup_tts_engine()
                break
            if command == 'r':
                success = toggle_recording()
                if success:
                    global current_speaker
                    current_speaker = DEFAULT_SPEAKER
                    process_audio(AUDIO_FILE)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        except KeyboardInterrupt:
            cleanup_tts_engine()
            break
    cleanup_tts_engine()

if __name__ == "__main__":
    main()

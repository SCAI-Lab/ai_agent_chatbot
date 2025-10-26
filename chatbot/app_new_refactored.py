"""AI-powered voice chatbot with personality analysis and long-term memory."""
import argparse
import json
import random
import time as time_module
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd
import torch

from modules.config import (
    DEFAULT_SPEAKER,
    DEFAULT_HISTORY_WINDOW,
    AUDIO_FILE,
    logger,
)
from modules.audio import TTSEngine, toggle_recording, cleanup_audio_file
from modules.database import init_db, store_personality_traits, fetch_user_data
from modules.llm import chat
from modules.memory import append_chat_to_cache, format_short_term_memory
from modules.personality import predict_personality, load_personality_model
from modules.speech import load_whisper_pipeline, transcribe_whisper
from modules.timing import timing, timing_context, clear_timings, print_timings


# Global state
current_speaker = DEFAULT_SPEAKER
preferences: dict[str, Any] = {}
history_window_size = DEFAULT_HISTORY_WINDOW


@dataclass
class ConversationState:
    """Conversation state for a speaker."""
    history: List[Dict[str, str]] = field(default_factory=list)


conversation_states: Dict[str, ConversationState] = {}


def get_conversation_state(speaker_key: str) -> ConversationState:
    """Get or create conversation state for a speaker.

    Args:
        speaker_key: Speaker identifier.

    Returns:
        ConversationState for the speaker.
    """
    state = conversation_states.get(speaker_key)
    if state is None:
        state = ConversationState()
        conversation_states[speaker_key] = state
    return state


@timing("greeting_tts")
def say_greeting(tts_engine: TTSEngine, speaker: str):
    """Say greeting message.

    Args:
        tts_engine: TTS engine instance
        speaker: Current speaker name
    """
    GREETINGS = [
        "Give me a sec to think about that.",
        "Let me process that real quick.",
        "Just a moment, I'm working on it.",
        "Let me figure this out for you.",
        "Hold on, I'll get right back to you.",
        "One moment while I put this together.",
    ]

    greeting = (
        f"Hello {speaker}. {random.choice(GREETINGS)}"
        if speaker
        else f"Sorry, I couldn't recognize you. {random.choice(GREETINGS)}"
    )

    tts_engine.say(greeting)


@timing("transcription")
def transcribe_audio_file(audio_file: str, pipe) -> str:
    """Transcribe audio file to text.

    Args:
        audio_file: Path to audio file
        pipe: Whisper pipeline

    Returns:
        Transcribed text

    Raises:
        ValueError: If transcription is empty
    """
    transcription = transcribe_whisper(audio_file, pipe)
    text = transcription.strip()

    if not text:
        raise ValueError("Transcription is empty")

    return text


@timing("personality_analysis")
def analyze_personality(text: str) -> pd.DataFrame:
    """Analyze personality from text.

    Args:
        text: Input text

    Returns:
        DataFrame with personality scores
    """
    predictions = predict_personality(text)
    return pd.DataFrame({"r": predictions, "theta": ["EXT", "NEU", "AGR", "CON", "OPN"]})


@timing("context_building")
def build_prompt_context(text: str, personality_df: pd.DataFrame, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build chat prompt with context.

    Args:
        text: User input text
        personality_df: Personality analysis dataframe
        history: Conversation history

    Returns:
        List of chat messages
    """
    global preferences, history_window_size

    recent_history = history[-2 * history_window_size:] if history_window_size > 0 else history

    personality_context = personality_df.to_string()
    preferences_context = json.dumps(preferences) if preferences else "None."
    memory_context = format_short_term_memory(recent_history)

    system_prompt = "\n\n".join([
        "You are Hackcelerate, a helpful healthcare assistant.",
        memory_context,
        "--# USER PERSONALITY TRAITS #--\nBig Five Personality traits inferred for the user (use them when necessary):\n" + personality_context + "\n--# END OF PERSONALITY TRAITS #--",
        "--# USER PREFERENCES #--\nKnown user preferences:\n" + preferences_context + "\n--# END OF PREFERENCES #--",
    ])

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def get_llm_response(chat_messages: List[Dict[str, str]], speaker: str) -> tuple:
    """Get LLM response.

    Note: Timing is handled inside llm.chat() for detailed TTFT metrics.

    Args:
        chat_messages: Chat messages
        speaker: Current speaker

    Returns:
        Tuple of (response_content, user_uuid)
    """
    import time
    from modules.timing import _timings

    start = time.perf_counter()
    response_content, user_uuid = chat(
        chat_messages,
        current_speaker=speaker,
        close_session=False,
        use_users=bool(speaker),
    )
    _timings['llm_total'] = time.perf_counter() - start

    if not response_content:
        raise ValueError("LLM response empty")

    return response_content, user_uuid


@timing("database_storage")
def save_conversation_data(speaker: str, predictions: List[float], user_uuid: str,
                          user_text: str, response: str, db_session):
    """Save conversation data to database.

    Args:
        speaker: Current speaker
        predictions: Personality predictions
        user_uuid: User UUID
        user_text: User input text
        response: Assistant response
        db_session: Database session
    """
    # Store personality traits
    store_personality_traits(speaker, predictions, db_session)

    # Cache conversation (with dummy durations for now)
    if user_uuid and response:
        append_chat_to_cache(
            user_uuid,
            speaker or DEFAULT_SPEAKER,
            user_text,
            response,
            0.0,  # speech_duration - handled separately
            0.0,  # llm_duration - handled separately
        )


@timing("response_tts")
def say_response(tts_engine: TTSEngine, response: str):
    """Speak the response.

    Args:
        tts_engine: TTS engine
        response: Response text
    """
    tts_engine.say(response)


def process_audio(audio_file: str, tts_engine: TTSEngine, db_session):
    """Process audio file through the full pipeline.

    Args:
        audio_file: Path to audio file.
        tts_engine: TTS engine instance.
        db_session: Database session.
    """
    global current_speaker

    # Clear previous session timing
    clear_timings()

    # Say greeting
    say_greeting(tts_engine, current_speaker)

    # Load Whisper pipeline
    with timing_context("whisper_loading"):
        use_gpu = torch.cuda.is_available()
        try:
            pipe = load_whisper_pipeline(use_gpu)
        except Exception as e:
            logger.error(f"Failed to load Whisper pipeline: {e}")
            return

    try:
        # Transcribe audio
        text = transcribe_audio_file(audio_file, pipe)
        print("Q:", text)

        # Analyze personality
        personality_df = analyze_personality(text)

        # Get conversation state
        speaker_key = current_speaker or "anonymous"
        state = get_conversation_state(speaker_key)
        history = state.history

        # Build context and get LLM response
        chat_messages = build_prompt_context(text, personality_df, history)
        response_content, user_uuid = get_llm_response(chat_messages, current_speaker)

        # Save to database
        predictions = personality_df["r"].tolist()
        save_conversation_data(current_speaker, predictions, user_uuid, text, response_content, db_session)

        # Update conversation history
        with timing_context("history_update"):
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": response_content})

            if history_window_size > 0:
                max_items = 2 * history_window_size
                if len(history) > max_items:
                    del history[:-max_items]

        # Print processing time (before TTS)
        from modules.timing import get_timings
        processing_timings = get_timings()
        processing_total = sum(processing_timings.values())
        print(f"\n[Processing completed in {processing_total:.4f}s]")

        # Speak response
        say_response(tts_engine, response_content)

        # Print full timing summary
        print_timings("Audio Processing Performance")

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")

    finally:
        # Cleanup
        with timing_context("cleanup"):
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cleanup_audio_file(audio_file)


def main():
    """Main program loop."""
    parser = argparse.ArgumentParser(description="Audio processing with personality analysis")
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help="Number of recent conversation rounds to retain for short-term memory (0 disables).",
    )
    args = parser.parse_args()

    global history_window_size
    history_window_size = max(0, args.history_window)

    # Initialize database
    Session = init_db()
    db_session = Session()

    # Initialize TTS
    tts_engine = TTSEngine()

    # Load personality model (one-time at startup)
    try:
        load_personality_model()
    except Exception as e:
        logger.error(f"Failed to load personality model: {e}")
        logger.warning("Continuing without personality analysis")

    print("Type 'r' to record, 'q' to quit.")

    try:
        while True:
            command = input().strip().lower()

            if command == 'q':
                break

            if command == 'r':
                success = toggle_recording()
                if success:
                    global current_speaker
                    current_speaker = DEFAULT_SPEAKER
                    process_audio(AUDIO_FILE, tts_engine, db_session)

    except KeyboardInterrupt:
        pass

    finally:
        tts_engine.cleanup()
        db_session.close()


if __name__ == "__main__":
    main()

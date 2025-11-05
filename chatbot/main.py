"""AI-powered voice chatbot with personality analysis and long-term memory."""
import argparse
import json
import random
import time as time_module
from dataclasses import dataclass, field
from typing import Any, Dict, List,Optional
import time
import pandas as pd
import torch

from modules.config import (
    DEFAULT_SPEAKER,
    DEFAULT_HISTORY_WINDOW,
    AUDIO_FILE,
    GREETING_MESSAGES,
    logger,
)
from modules.audio import TTSEngine, cleanup_audio_file, select_input_device, record_audio
from modules.database import init_db, store_personality_traits
from modules.llm import chat
from modules.memory import append_chat_to_cache, format_short_term_memory
from modules.personality import predict_personality, load_personality_model
from modules.speech import load_whisper_pipeline, transcribe_whisper
from modules.emotion import load_emotion_model, predict_emotion
from modules.timing import timing, timing_context, clear_timings, print_timings, _record_timing


@dataclass
class ConversationState:
    """Conversation state for a speaker."""
    history: List[Dict[str, str]] = field(default_factory=list)
    greeting_played: bool = False


@dataclass
class ApplicationState:
    """Global application state."""
    current_speaker: str = DEFAULT_SPEAKER
    preferences: Dict[str, Any] = field(default_factory=dict)
    history_window_size: int = DEFAULT_HISTORY_WINDOW
    whisper_pipeline: Any = None
    selected_device_index: Optional[int] = None
    debug_mode: bool = False
    conversation_states: Dict[str, ConversationState] = field(default_factory=dict)

    def get_conversation_state(self, speaker_key: str) -> ConversationState:
        """Get or create conversation state for a speaker.

        Args:
            speaker_key: Speaker identifier.

        Returns:
            ConversationState for the speaker.
        """
        if speaker_key not in self.conversation_states:
            self.conversation_states[speaker_key] = ConversationState()
        return self.conversation_states[speaker_key]


@timing("greeting_tts")
def say_greeting(tts_engine: TTSEngine, speaker: str):
    """Say greeting message.

    Args:
        tts_engine: TTS engine instance
        speaker: Current speaker name
    """
    greeting = (
        f"Hello {speaker}. {random.choice(GREETING_MESSAGES)}"
        if speaker
        else f"Sorry, I couldn't recognize you. {random.choice(GREETING_MESSAGES)}"
    )
    tts_engine.say(greeting)


@timing("transcription")
def transcribe_audio_file(audio_file: str, whisper_pipeline: Any) -> str:
    """Transcribe audio file to text.

    Args:
        audio_file: Path to audio file
        whisper_pipeline: Whisper pipeline instance

    Returns:
        Transcribed text

    Raises:
        ValueError: If transcription is empty
    """
    transcription = transcribe_whisper(audio_file, whisper_pipeline)
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


@timing("emotion_analysis")
def analyze_emotion(audio_file: str) -> Dict[str, float]:
    """Analyze emotion from audio file.

    Args:
        audio_file: Path to audio file

    Returns:
        Dictionary with emotion probabilities
    """
    try:
        return predict_emotion(audio_file)
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        # Return uniform distribution on error
        return {emotion: 1.0/7 for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']}


def build_prompt_context(
    text: str,
    personality_df: pd.DataFrame,
    emotion_dict: Dict[str, float],
    history: List[Dict[str, str]],
    preferences: Dict[str, Any],
    history_window_size: int,
) -> List[Dict[str, str]]:
    """Build chat prompt with context.

    Args:
        text: User input text
        personality_df: Personality analysis dataframe
        emotion_dict: Emotion probabilities (already normalized)
        history: Conversation history
        preferences: User preferences
        history_window_size: Number of conversation rounds to include

    Returns:
        List of chat messages
    """

    overall_start = time.perf_counter()

    # Extract recent history
    recent_history = history[-2 * history_window_size:] if history_window_size > 0 else history

    # Format contexts
    personality_context = personality_df.to_string()
    preferences_context = json.dumps(preferences) if preferences else "None."
    memory_context = format_short_term_memory(recent_history)

    # Format emotion context
    sorted_emotions = sorted(emotion_dict.items(), key=lambda item: item[1], reverse=True)
    dominant_emotion, emotion_confidence = sorted_emotions[0]
    emotion_lines = "\n".join([f"- {label}: {prob:.4f}" for label, prob in sorted_emotions])

    emotion_context = (
        f"Dominant emotion: {dominant_emotion} (confidence: {emotion_confidence:.4f})\n"
        f"All emotions:\n"
        f"{emotion_lines}"
    )

    # Build system prompt
    system_prompt = "\n\n".join([
        "You are Hackcelerate, a helpful healthcare assistant.",
        memory_context,
        f"--# USER PERSONALITY TRAITS #--\nBig Five Personality traits inferred for the user (use them when necessary):\n{personality_context}\n--# END OF PERSONALITY TRAITS #--",
        f"--# USER EMOTIONAL STATE #--\nEmotional state detected from voice (use for empathetic responses):\n{emotion_context}\n--# END OF EMOTIONAL STATE #--",
        f"--# USER PREFERENCES #--\nKnown user preferences:\n{preferences_context}\n--# END OF PREFERENCES #--",
    ])

    # Record timing
    _record_timing("memory_and_context", time.perf_counter() - overall_start)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def get_llm_response(
    chat_messages: List[Dict[str, str]],
    speaker: str,
    debug_mode: bool = False,
) -> tuple:
    """Get LLM response.

    Note: Timing is handled inside llm.chat() for detailed TTFT metrics.

    Args:
        chat_messages: Chat messages
        speaker: Current speaker
        debug_mode: Enable verbose prompt logging

    Returns:
        Tuple of (response_content, user_uuid)
    """
    start = time.perf_counter()
    response_content, user_uuid = chat(
        chat_messages,
        current_speaker=speaker,
        close_session=False,
        use_users=bool(speaker),
        debug=debug_mode,
    )
    _record_timing('llm_total', time.perf_counter() - start)

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


def process_audio(
    audio_file: str,
    tts_engine: TTSEngine,
    db_session,
    app_state: ApplicationState,
):
    """Process audio file through the full pipeline.

    Args:
        audio_file: Path to audio file.
        tts_engine: TTS engine instance.
        db_session: Database session.
        app_state: Application state object.
    """
    # Clear previous session timing
    clear_timings()

    # Get conversation state
    speaker_key = app_state.current_speaker or "anonymous"
    state = app_state.get_conversation_state(speaker_key)

    # Say greeting only once per speaker/session
    if not state.greeting_played:
        say_greeting(tts_engine, app_state.current_speaker)
        state.greeting_played = True

    try:
        # Step 1: Transcribe audio first (priority)
        text = transcribe_audio_file(audio_file, app_state.whisper_pipeline)
        print("Q:", text)

        # Step 2: Parallel emotion analysis and personality analysis
        emotion_dict = analyze_emotion(audio_file)
        personality_df = analyze_personality(text)

        # Build context and get LLM response
        chat_messages = build_prompt_context(
            text,
            personality_df,
            emotion_dict,
            state.history,
            app_state.preferences,
            app_state.history_window_size,
        )
        response_content, user_uuid = get_llm_response(
            chat_messages,
            app_state.current_speaker,
            app_state.debug_mode,
        )

        # Save to database
        predictions = personality_df["r"].tolist()
        save_conversation_data(
            app_state.current_speaker,
            predictions,
            user_uuid,
            text,
            response_content,
            db_session,
        )

        # Update conversation history
        with timing_context("history_update"):
            state.history.append({"role": "user", "content": text})
            state.history.append({"role": "assistant", "content": response_content})

            if app_state.history_window_size > 0:
                max_items = 2 * app_state.history_window_size
                if len(state.history) > max_items:
                    del state.history[:-max_items]

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
        cleanup_audio_file(audio_file)


def print_startup_timings(timings: Dict[str, float]):
    """Print initialization timing summary at startup.

    Args:
        timings: Dictionary of initialization step names and durations
    """
    print("\n" + "=" * 60)
    print("System Initialization")
    print("=" * 60)
    print(f"{'Component':<40} {'Duration':>10}")
    print("-" * 60)

    total = 0.0
    for name, duration in timings.items():
        print(f"{name:<40} {duration:>9.4f}s")
        total += duration

    print("-" * 60)
    print(f"{'TOTAL INITIALIZATION TIME':<40} {total:>9.4f}s")
    print("=" * 60)
    print()


def main():
    """Main program loop."""
    parser = argparse.ArgumentParser(description="Audio processing with personality analysis")
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help="Number of recent conversation rounds to retain for short-term memory (0 disables).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the complete prompt payload sent to the LLM each round.",
    )
    args = parser.parse_args()

    # Initialize application state
    app_state = ApplicationState(
        history_window_size=max(0, args.history_window),
        debug_mode=args.debug,
    )

    # Track initialization timings
    init_timings = {}

    # Initialize database
    start = time_module.perf_counter()
    Session = init_db()
    db_session = Session()
    init_timings["Database initialization"] = time_module.perf_counter() - start

    # Initialize TTS
    start = time_module.perf_counter()
    tts_engine = TTSEngine()
    init_timings["TTS engine initialization"] = time_module.perf_counter() - start

    # Load personality model (one-time at startup)
    start = time_module.perf_counter()
    try:
        load_personality_model()
        init_timings["Big5 personality model & tokenizer"] = time_module.perf_counter() - start
    except Exception as e:
        init_timings["Big5 personality model & tokenizer (FAILED)"] = time_module.perf_counter() - start
        logger.error(f"Failed to load personality model: {e}")
        logger.warning("Continuing without personality analysis")

    # Load emotion recognition model (one-time at startup)
    start = time_module.perf_counter()
    try:
        load_emotion_model()
        init_timings["Wav2Vec2 emotion recognition model"] = time_module.perf_counter() - start
    except Exception as e:
        init_timings["Wav2Vec2 emotion recognition model (FAILED)"] = time_module.perf_counter() - start
        logger.error(f"Failed to load emotion model: {e}")
        logger.warning("Continuing without emotion analysis")

    # Load Whisper pipeline (one-time at startup)
    use_gpu = torch.cuda.is_available()
    start = time_module.perf_counter()
    try:
        logger.info(f"Loading Whisper pipeline (GPU: {use_gpu})...")
        app_state.whisper_pipeline = load_whisper_pipeline(use_gpu)
        logger.info("Whisper pipeline loaded successfully")
        init_timings["Whisper speech-to-text model"] = time_module.perf_counter() - start
    except Exception as e:
        init_timings["Whisper speech-to-text model (FAILED)"] = time_module.perf_counter() - start
        logger.error(f"Failed to load Whisper pipeline: {e}")
        return

    # Test Ollama connection (we can't control loading, but we can test it)
    start = time_module.perf_counter()
    try:
        from modules.llm import client
        # Simple test to see if Ollama is responsive
        models = client.models.list()
        init_timings["Ollama LLM connection check"] = time_module.perf_counter() - start
    except Exception as e:
        init_timings["Ollama LLM connection check (FAILED)"] = time_module.perf_counter() - start
        logger.warning(f"Ollama connection test failed: {e}")

    # Print startup timings
    print_startup_timings(init_timings)

    print("Type 'r' to record, 'q' to quit.")

    try:
        while True:
            command = input().strip().lower()

            if command == 'q':
                break

            if command == 'r':
                # Select device only on first recording
                if app_state.selected_device_index is None:
                    print("\nFirst time setup - please select your audio input device:")
                    app_state.selected_device_index = select_input_device()
                    print(f"Device selected. This will be used for all recordings in this session.\n")

                # Use the remembered device for recording
                success = record_audio(device_index=app_state.selected_device_index)
                if success:
                    app_state.current_speaker = DEFAULT_SPEAKER
                    process_audio(AUDIO_FILE, tts_engine, db_session, app_state)

    except KeyboardInterrupt:
        pass

    finally:
        # Cleanup resources
        tts_engine.cleanup()
        db_session.close()

        # Cleanup Whisper pipeline
        if app_state.whisper_pipeline is not None:
            del app_state.whisper_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

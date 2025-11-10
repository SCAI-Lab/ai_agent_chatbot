"""AI-powered voice chatbot with personality analysis and long-term memory."""
import argparse
import json
import random
import time as time_module
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import numpy as np
import pandas as pd
import torch

from modules.config import (
    DEFAULT_SPEAKER,
    DEFAULT_HISTORY_WINDOW,
    AUDIO_FILE,
    GREETING_MESSAGES,
    SPEECH_EMOTION_WEIGHT,
    TEXT_EMOTION_WEIGHT,
    logger,
)
from modules.audio import TTSEngine, cleanup_audio_file, select_input_device, record_audio
from modules.database import init_db, store_personality_traits
from modules.llm import chat
from modules.memory import append_chat_to_cache, format_short_term_memory, flush_cache_to_disk
from modules.personality import predict_personality, load_personality_model
from modules.speech2text import load_whisper_pipeline, transcribe_whisper
from modules.speech2emotion import load_emotion_model, predict_emotion
from modules.text2emotion import load_text_emotion_model, predict_text_emotion, TEXT_EMOTION_LABELS
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
    speech_emotion_weight: float = SPEECH_EMOTION_WEIGHT
    text_emotion_weight: float = TEXT_EMOTION_WEIGHT

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
    return pd.DataFrame({
        "r": predictions,
        "theta": ["Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness"]
    })


@timing("speech2emotion_analysis")
def analyze_speech_emotion(audio_file: str, return_logits: bool = False) -> Dict[str, float]:
    """Analyze emotion from speech audio.

    Args:
        audio_file: Path to audio file
        return_logits: If True, return logits instead of probabilities

    Returns:
        Dictionary with emotion logits or probabilities
    """
    try:
        return predict_emotion(audio_file, return_logits=return_logits)
    except Exception as e:
        logger.error(f"Speech emotion analysis failed: {e}")
        if return_logits:
            return {label: 0.0 for label in TEXT_EMOTION_LABELS}
        else:
            uniform = 1.0 / len(TEXT_EMOTION_LABELS)
            return {label: uniform for label in TEXT_EMOTION_LABELS}


@timing("text2emotion_analysis")
def analyze_text_emotion(text: str, return_logits: bool = False) -> Dict[str, float]:
    """Analyze emotion from text content.

    Args:
        text: Input text
        return_logits: If True, return logits instead of probabilities

    Returns:
        Dictionary with emotion logits or probabilities
    """
    try:
        return predict_text_emotion(text, return_logits=return_logits)
    except Exception as e:
        logger.error(f"Text emotion analysis failed: {e}")
        if return_logits:
            return {label: 0.0 for label in TEXT_EMOTION_LABELS}
        else:
            uniform = 1.0 / len(TEXT_EMOTION_LABELS)
            return {label: uniform for label in TEXT_EMOTION_LABELS}


def fuse_emotions(
    speech_emotion: Dict[str, float],
    text_emotion: Dict[str, float],
    speech_weight: float,
    text_weight: float,
) -> Dict[str, float]:
    """Fuse speech and text emotions using probability averaging: p = λ * p_speech + (1-λ) * p_text

    Args:
        speech_emotion: Speech emotion logits
        text_emotion: Text emotion logits
        speech_weight: Weight for speech emotion (0.0-1.0)
        text_weight: Weight for text emotion (0.0-1.0)

    Returns:
        Fused emotion probabilities (normalized, sum=1.0)
    """
    # Normalize weights
    total_weight = speech_weight + text_weight
    if total_weight == 0:
        # If both weights are 0, return uniform distribution
        uniform = 1.0 / len(TEXT_EMOTION_LABELS)
        return {label: uniform for label in TEXT_EMOTION_LABELS}

    lambda_speech = speech_weight / total_weight
    lambda_text = text_weight / total_weight

    # Convert logits to probabilities via softmax
    speech_logits_array = np.array([speech_emotion[label] for label in TEXT_EMOTION_LABELS])
    speech_exp = np.exp(speech_logits_array - np.max(speech_logits_array))
    speech_probs = speech_exp / np.sum(speech_exp)

    text_logits_array = np.array([text_emotion[label] for label in TEXT_EMOTION_LABELS])
    text_exp = np.exp(text_logits_array - np.max(text_logits_array))
    text_probs = text_exp / np.sum(text_exp)

    # Weighted average
    fused_probs = lambda_speech * speech_probs + lambda_text * text_probs

    return {TEXT_EMOTION_LABELS[i]: float(fused_probs[i]) for i in range(len(TEXT_EMOTION_LABELS))}


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
    # Extract recent history
    recent_history = history[-2 * history_window_size:] if history_window_size > 0 else history

    # Format contexts
    personality_traits = ", ".join([f"{row['theta']}: {row['r']:.2f}" for _, row in personality_df.iterrows()])
    personality_context = f"user's personality: {personality_traits}"
    preferences_context = json.dumps(preferences) if preferences else "None."
    memory_context = format_short_term_memory(recent_history)

    # Format emotion context
    emotion_lines = ", ".join([f"{label}: {prob:.2f}" for label, prob in sorted(emotion_dict.items())])
    emotion_context = f"user's detected emotion: {emotion_lines}"

    # Build system prompt
    system_prompt = "\n\n".join([
        "You are Hackcelerate, a helpful healthcare assistant.",
        memory_context,
        f"--# USER PERSONALITY TRAITS #--\n{personality_context}\n--# END OF PERSONALITY TRAITS #--",
        f"--# USER EMOTIONAL STATE #--\n{emotion_context}\n--# END OF EMOTIONAL STATE #--",
        f"--# USER PREFERENCES #--\nKnown user preferences:\n{preferences_context}\n--# END OF PREFERENCES #--",
    ])

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
        # Start measuring total processing time (from audio input to TTS output)
        total_processing_start = time.perf_counter()

        # Step 1: Start Whisper transcription AND speech emotion analysis in parallel
        # This overlaps the two most time-consuming tasks
        parallel_audio_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Both work on the same audio file simultaneously
            transcription_future = executor.submit(
                transcribe_audio_file, audio_file, app_state.whisper_pipeline
            )
            speech_emotion_future = executor.submit(
                analyze_speech_emotion, audio_file, return_logits=True
            )

            # Wait for transcription to complete (needed for text-based analysis)
            text = transcription_future.result()
            print("Q:", text)

            # Start text-based analysis while speech emotion may still be running
            text_emotion_future = executor.submit(
                analyze_text_emotion, text, return_logits=True
            )
            personality_future = executor.submit(analyze_personality, text)

            # Wait for all results
            speech_emotion = speech_emotion_future.result()
            text_emotion = text_emotion_future.result()
            personality_df = personality_future.result()

        # Record parallel block wall clock time
        _record_timing("[Parallel] Audio processing (Whisper + speech2emotion + text2emotion + personality)",
                      time.perf_counter() - parallel_audio_start)

        # Step 3: Fuse emotions
        emotion_dict = fuse_emotions(
            speech_emotion,
            text_emotion,
            app_state.speech_emotion_weight,
            app_state.text_emotion_weight,
        )

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
        state.history.append({"role": "user", "content": text})
        state.history.append({"role": "assistant", "content": response_content})

        if app_state.history_window_size > 0:
            max_items = 2 * app_state.history_window_size
            if len(state.history) > max_items:
                del state.history[:-max_items]

        # Calculate total wall clock time from audio input to TTS start
        total_processing_time = time.perf_counter() - total_processing_start
        print(f"\n[Processing completed in {total_processing_time:.4f}s (time to first speech)]")

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

        # Only add to total if not a parallel sub-task (sub-tasks start with "  ├─")
        if not name.startswith("  ├─"):
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
    parser.add_argument(
        "--speech-emotion-weight",
        type=float,
        default=SPEECH_EMOTION_WEIGHT,
        help=f"Weight for speech-based emotion analysis (0.0-1.0, default: {SPEECH_EMOTION_WEIGHT}).",
    )
    parser.add_argument(
        "--text-emotion-weight",
        type=float,
        default=TEXT_EMOTION_WEIGHT,
        help=f"Weight for text-based emotion analysis (0.0-1.0, default: {TEXT_EMOTION_WEIGHT}).",
    )
    args = parser.parse_args()

    # Initialize application state
    app_state = ApplicationState(
        history_window_size=max(0, args.history_window),
        debug_mode=args.debug,
        speech_emotion_weight=args.speech_emotion_weight,
        text_emotion_weight=args.text_emotion_weight,
    )

    # Track initialization timings
    init_timings = {}

    # Sequential: Database initialization (fast)
    start = time_module.perf_counter()
    Session = init_db()
    db_session = Session()
    init_timings["Database initialization"] = time_module.perf_counter() - start

    # Sequential: TTS engine initialization (fast)
    start = time_module.perf_counter()
    tts_engine = TTSEngine()
    init_timings["TTS engine initialization"] = time_module.perf_counter() - start

    # Sequential: Load personality model (has issues with parallel loading)
    start = time_module.perf_counter()
    try:
        load_personality_model()
        init_timings["Big5 personality model & tokenizer"] = time_module.perf_counter() - start
    except Exception as e:
        init_timings["Big5 personality model & tokenizer (FAILED)"] = time_module.perf_counter() - start
        logger.error(f"Failed to load personality model: {e}")
        logger.warning("Continuing without personality analysis")

    # Parallel: Load emotion models only (speech2emotion + text2emotion)
    parallel_tasks = []
    parallel_task_timings = {}

    def speech_emotion_task():
        """Load speech emotion model."""
        task_start = time_module.perf_counter()
        try:
            load_emotion_model()
            return True, time_module.perf_counter() - task_start
        except Exception as e:
            logger.error(f"Failed to load speech emotion model: {e}")
            logger.warning("Continuing without speech emotion analysis")
            return False, time_module.perf_counter() - task_start

    def text_emotion_task():
        """Load text emotion model."""
        task_start = time_module.perf_counter()
        try:
            load_text_emotion_model()
            return True, time_module.perf_counter() - task_start
        except Exception as e:
            logger.error(f"Failed to load text emotion model: {e}")
            logger.warning("Continuing without text emotion analysis")
            return False, time_module.perf_counter() - task_start

    # Conditionally load emotion models based on weights
    if app_state.speech_emotion_weight > 0:
        parallel_tasks.append(("Speech2Emotion recognition model", speech_emotion_task))

    if app_state.text_emotion_weight > 0:
        parallel_tasks.append(("Text2Emotion DeBERTa model", text_emotion_task))

    # Execute parallel tasks and measure wall clock time
    if parallel_tasks:
        parallel_block_start = time_module.perf_counter()
        with ThreadPoolExecutor(max_workers=len(parallel_tasks)) as executor:
            futures = {name: executor.submit(task) for name, task in parallel_tasks}

            # Wait for all tasks to complete and collect individual timings
            for name, future in futures.items():
                try:
                    success, duration = future.result()
                    if success:
                        parallel_task_timings[name] = duration
                    else:
                        parallel_task_timings[f"{name} (FAILED)"] = duration
                except Exception as e:
                    parallel_task_timings[f"{name} (FAILED)"] = 0.0
                    logger.error(f"Failed to load {name}: {e}")

        # Record actual wall clock time for the parallel block
        parallel_block_duration = time_module.perf_counter() - parallel_block_start
        init_timings["[Parallel] Emotion models (speech + text)"] = parallel_block_duration

        # Also store individual timings for detailed view
        for name, duration in parallel_task_timings.items():
            init_timings[f"  ├─ {name}"] = duration

    # Sequential: Load Whisper pipeline (large model, avoid GPU conflicts)
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

    # Sequential: Test Ollama connection (fast check)
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
        # Flush memory cache to disk before exit
        flush_cache_to_disk()

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

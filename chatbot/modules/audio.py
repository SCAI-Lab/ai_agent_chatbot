"""Audio recording and text-to-speech module."""
import logging
import os
import threading
from typing import Optional

import numpy as np
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wavfile

from .config import (
    SAMPLE_RATE,
    RECORD_DURATION,
    AUDIO_FILE,
    TTS_RATE,
    TTS_VOLUME,
    AUDIO_TIMEOUT_MARGIN,
    AUDIO_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-speech engine wrapper."""

    def __init__(self):
        """Initialize TTS engine."""
        self.engine: Optional[pyttsx3.Engine] = None
        self._init_engine()

    def _init_engine(self):
        """Initialize pyttsx3 engine with configuration."""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', TTS_RATE)
            self.engine.setProperty('volume', TTS_VOLUME)
            voices = self.engine.getProperty('voices')
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[0].id)
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None

    def say(self, text: str) -> bool:
        """Speak text using TTS.

        Args:
            text: Text to speak.

        Returns:
            True if successful, False otherwise.
        """
        if not self.engine:
            logger.error("TTS engine not initialized.")
            return False

        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
            return False

    def cleanup(self):
        """Clean up TTS engine resources."""
        if self.engine:
            try:
                self.engine.stop()
                if getattr(self.engine, "_inLoop", False):
                    self.engine.endLoop()
            except Exception as e:
                logger.error(f"Failed to clean up TTS engine: {e}")
            self.engine = None


def _safe_stop_recording():
    """Safely stop sounddevice recording with error handling."""
    try:
        sd.stop()
        logger.info("Recording stopped successfully")
    except Exception as e:
        logger.error(f"Failed to stop recording: {e}")


def select_input_device() -> Optional[int]:
    """Prompt user to select an audio input device.

    Returns:
        Device index or None for default device.
    """
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


def record_audio(
    device_index: Optional[int] = None,
    duration: int = RECORD_DURATION,
    sample_rate: int = SAMPLE_RATE,
    output_file: str = AUDIO_FILE,
    timeout_margin: float = AUDIO_TIMEOUT_MARGIN,
) -> bool:
    """Record audio from microphone.

    Args:
        device_index: Audio device index (None for default).
        duration: Recording duration in seconds.
        sample_rate: Audio sample rate.
        output_file: Output WAV file path.
        timeout_margin: Extra seconds to wait beyond duration before timeout.

    Returns:
        True if recording successful, False otherwise.
    """
    try:
        # Log device information for debugging
        if device_index is not None:
            try:
                device_info = sd.query_devices(device_index)
                logger.info(f"Using device {device_index}: {device_info['name']}")
                logger.info(f"Device max input channels: {device_info['max_input_channels']}")
                logger.info(f"Device default samplerate: {device_info['default_samplerate']}")
            except Exception as e:
                logger.warning(f"Could not query device info: {e}")

        print(f"Recording for {duration} seconds...")

        # Start recording
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_index,
            blocking=False
        )

        # Wait with timeout to prevent infinite hanging
        timeout_seconds = duration + timeout_margin
        wait_completed = threading.Event()

        def wait_for_recording():
            try:
                sd.wait()
                wait_completed.set()
            except Exception as e:
                logger.error(f"Error during sd.wait(): {e}")
                wait_completed.set()

        wait_thread = threading.Thread(target=wait_for_recording, daemon=True)
        wait_thread.start()

        # Wait for completion or timeout
        if not wait_completed.wait(timeout=timeout_seconds):
            logger.error(f"Recording timeout after {timeout_seconds}s. Attempting to stop...")

            # Try to stop recording in a non-blocking way
            stop_thread = threading.Thread(target=lambda: _safe_stop_recording(), daemon=True)
            stop_thread.start()
            stop_thread.join(timeout=1.0)  # Wait max 1 second for stop

            if stop_thread.is_alive():
                logger.error("sd.stop() is also hanging. Giving up on this recording.")

            return False

        # Validate audio data
        if audio_data is None:
            logger.error("No audio data recorded (audio_data is None).")
            return False

        if not np.any(audio_data):
            logger.warning("Audio data is all zeros - microphone may not be working.")
            # Still save it, might be useful for debugging

        # Check for NaN or inf values
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            logger.error("Audio data contains NaN or inf values.")
            return False

        # Save audio file
        wavfile.write(output_file, sample_rate, audio_data)
        print(f"Audio recording saved to {output_file}")

        # Log audio statistics for debugging
        logger.info(f"Audio stats - Shape: {audio_data.shape}, "
                   f"Min: {np.min(audio_data):.4f}, Max: {np.max(audio_data):.4f}, "
                   f"Mean: {np.mean(audio_data):.4f}")

        return True

    except Exception as e:
        logger.error(f"Audio recording failed: {e}", exc_info=True)
        # Try to stop in a non-blocking way
        stop_thread = threading.Thread(target=_safe_stop_recording, daemon=True)
        stop_thread.start()
        stop_thread.join(timeout=0.5)
        return False


def validate_device_capabilities(device_index: Optional[int], sample_rate: int = SAMPLE_RATE) -> bool:
    """Validate that the device supports required recording configuration.

    Args:
        device_index: Audio device index (None for default).
        sample_rate: Desired sample rate.

    Returns:
        True if device is compatible, False otherwise.
    """
    try:
        device_info = sd.query_devices(device_index, 'input')

        # Check if device has input channels
        if device_info['max_input_channels'] < 1:
            logger.error(f"Device has no input channels: {device_info['name']}")
            return False

        # Check if sample rate is supported (with some tolerance)
        default_sr = device_info['default_samplerate']
        if abs(default_sr - sample_rate) > 1000:
            logger.warning(f"Device default samplerate ({default_sr}) differs from requested ({sample_rate})")
            # Don't fail, just warn - sounddevice can resample

        logger.info(f"Device validation passed for: {device_info['name']}")
        return True

    except Exception as e:
        logger.error(f"Failed to validate device: {e}")
        return False


def toggle_recording(max_retries: int = AUDIO_MAX_RETRIES) -> bool:
    """Interactive audio recording with device selection and retry logic.

    Args:
        max_retries: Maximum number of retry attempts on failure.

    Returns:
        True if recording successful, False otherwise.
    """
    device_index = select_input_device()

    if device_index is None and not any(d['max_input_channels'] > 0 for d in sd.query_devices()):
        logger.error("No valid input device. Aborting recording.")
        return False

    # Validate device capabilities
    if device_index is not None:
        if not validate_device_capabilities(device_index):
            logger.warning("Device validation failed, but attempting to proceed anyway...")

    # Attempt recording with retries
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.info(f"Retry attempt {attempt}/{max_retries}...")
            print(f"Retrying recording (attempt {attempt}/{max_retries})...")

        success = record_audio(device_index)

        if success:
            return True

        if attempt < max_retries:
            logger.warning("Recording failed, will retry...")
            # Small delay before retry
            import time
            time.sleep(0.5)

    logger.error(f"Recording failed after {max_retries + 1} attempts.")
    return False


def cleanup_audio_file(audio_file: str = AUDIO_FILE):
    """Remove temporary audio file.

    Args:
        audio_file: Path to audio file to delete.
    """
    if os.path.exists(audio_file):
        try:
            os.remove(audio_file)
        except Exception as e:
            logger.error(f"Failed to remove audio file {audio_file}: {e}")

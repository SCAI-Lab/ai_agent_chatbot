"""Speech emotion recognition module using Wav2Vec2."""
import os
import logging
import torch
import numpy as np
from typing import Dict
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import soundfile as sf
from scipy import signal

# Set cache directory to SSD for model downloads
os.environ['HF_HOME'] = '/mnt/ssd/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/ssd/huggingface'

logger = logging.getLogger(__name__)

# Global model components (loaded once at module import)
_model = None
_feature_extractor = None
_device = None

# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def load_emotion_model():
    """Load emotion recognition model from HuggingFace with optimizations."""
    global _model, _feature_extractor, _device

    if _model is not None:
        return

    try:
        logger.info("Loading emotion recognition model...")

        model_name = "r-f/wav2vec-english-speech-emotion-recognition"
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        _model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = _model.to(_device)
        _model.eval()

        # Apply optimizations for CUDA (similar to speech.py Whisper loading)
        if torch.cuda.is_available():
            # Use half precision (fp16) for faster inference
            _model = _model.half()
            logger.info(f"Using fp16 precision on {_device}")

        logger.info(f"Emotion model loaded on {_device} (dtype: {next(_model.parameters()).dtype})")

    except Exception as e:
        logger.error(f"Failed to load emotion model: {e}")
        raise


def predict_emotion(audio_path: str) -> Dict[str, float]:
    """Predict emotion probabilities from audio file.

    Args:
        audio_path: Path to audio file (WAV format recommended).

    Returns:
        Dictionary with emotion names and probabilities (0-1 range):
        {'angry': 0.1, 'disgust': 0.05, 'fear': 0.1, 'happy': 0.3,
         'neutral': 0.2, 'sad': 0.15, 'surprise': 0.1}
    """
    if _model is None:
        load_emotion_model()

    try:
        # Fast audio loading with soundfile
        audio, orig_sr = sf.read(audio_path, dtype='float32')

        # Resample to 16kHz if needed (using scipy for speed)
        target_sr = 16000
        if orig_sr != target_sr:
            num_samples = int(len(audio) * target_sr / orig_sr)
            audio = signal.resample(audio, num_samples)

        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Extract features and run inference
        inputs = _feature_extractor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)

        # Convert inputs to fp16 if model is fp16
        if torch.cuda.is_available() and next(_model.parameters()).dtype == torch.float16:
            inputs = {k: v.to(_device).half() if v.dtype == torch.float32 else v.to(_device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = _model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().float().numpy()

        # Return as dictionary
        return {EMOTION_LABELS[i]: float(probabilities[i]) for i in range(len(EMOTION_LABELS))}

    except Exception as e:
        logger.error(f"Emotion prediction failed: {e}")
        # Return uniform distribution on error
        return {label: 1.0/7 for label in EMOTION_LABELS}

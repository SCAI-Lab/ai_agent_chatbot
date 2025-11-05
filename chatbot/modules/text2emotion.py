"""Text emotion classification using DeBERTa-v3-Large.

Model: https://huggingface.co/Tanneru/Emotion-Classification-DeBERTa-v3-Large
Predicts 7 emotion classes: anger, disgust, fear, happy, neutral, sad, surprise.
"""
import logging
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# Model from Hugging Face
MODEL_NAME = "Tanneru/Emotion-Classification-DeBERTa-v3-Large"

# Emotion labels (7 classes)
TEXT_EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

_text_model: Optional[AutoModelForSequenceClassification] = None
_text_tokenizer: Optional[AutoTokenizer] = None
_text_device: Optional[torch.device] = None


def load_text_emotion_model() -> None:
    """Load the DeBERTa-v3-Large text emotion classification model."""
    global _text_model, _text_tokenizer, _text_device

    if _text_model is not None:
        return

    logger.info("Loading DeBERTa-v3-Large text emotion classification model...")

    _text_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use fast tokenizer if available (doesn't require sentencepiece slow tokenizer)
    try:
        _text_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        logger.info("Using fast tokenizer")
    except Exception:
        # Fallback to slow tokenizer if fast one is not available
        logger.info("Fast tokenizer not available, using slow tokenizer")
        _text_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # Load model (PyTorch 2.7+ compatible - no low_cpu_mem_usage)
    _text_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=False
    )
    _text_model = _text_model.to(_text_device)
    _text_model.eval()

    logger.info("Text emotion model loaded on %s", _text_device)


def predict_text_emotion(text: str, return_logits: bool = False) -> Dict[str, float]:
    """Predict emotion probabilities or logits for the given text.

    Args:
        text: Input text to analyze
        return_logits: If True, return logits instead of probabilities

    Returns:
        Dictionary mapping emotion labels to logits or probabilities
    """
    if _text_model is None or _text_tokenizer is None or _text_device is None:
        load_text_emotion_model()

    try:
        # Tokenize input
        inputs = _text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(_text_device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = _text_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # Convert to dictionary
        if return_logits:
            logits_np = logits.squeeze(0).cpu().numpy()
            return {TEXT_EMOTION_LABELS[i]: float(logits_np[i]) for i in range(len(TEXT_EMOTION_LABELS))}
        else:
            probs = probabilities.squeeze(0).cpu().numpy()
            return {TEXT_EMOTION_LABELS[i]: float(probs[i]) for i in range(len(TEXT_EMOTION_LABELS))}

    except Exception as exc:
        logger.error("Text emotion prediction failed: %s", exc)
        if return_logits:
            return {label: 0.0 for label in TEXT_EMOTION_LABELS}
        else:
            uniform = 1.0 / len(TEXT_EMOTION_LABELS)
            return {label: uniform for label in TEXT_EMOTION_LABELS}


def get_dominant_emotion(text: str) -> str:
    """Get the dominant emotion label for the given text.

    Args:
        text: Input text to analyze

    Returns:
        The emotion label with highest probability
    """
    emotion_probs = predict_text_emotion(text)
    return max(emotion_probs.items(), key=lambda x: x[1])[0]

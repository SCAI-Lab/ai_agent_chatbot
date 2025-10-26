"""Big Five personality analysis module using BERT."""
import logging
import torch
from typing import List, Dict
from transformers import BertTokenizer, BertForSequenceClassification

logger = logging.getLogger(__name__)


# Global model and tokenizer (loaded once at module import)
_model = None
_tokenizer = None
_device = None


def load_personality_model():
    """Load the BERT personality model and tokenizer.

    This should be called once at application startup.
    """
    global _model, _tokenizer, _device

    if _model is not None:
        logger.info("Personality model already loaded")
        return

    try:
        logger.info("Loading Big5 personality model...")

        # Load tokenizer and model
        model_name = "Minej/bert-base-personality"
        _tokenizer = BertTokenizer.from_pretrained(model_name)
        _model = BertForSequenceClassification.from_pretrained(model_name)

        logger.info(f"Model: {model_name}")
        logger.info(f"Tokenizer vocab size: {_tokenizer.vocab_size}")
        logger.info(f"Model parameters: {sum(p.numel() for p in _model.parameters()) / 1e6:.1f}M")

        # Set to evaluation mode
        _model.eval()

        # Use GPU if available
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = _model.to(_device)

        # Enable optimizations for GPU
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Device: {device_name} (CUDA)")

            # Warmup GPU
            warmup_text = "This is a warmup"
            inputs = _tokenizer(warmup_text, truncation=True, padding=True, return_tensors="pt")
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = _model(**inputs)
            logger.info("GPU warmup complete")
        else:
            logger.info("Device: CPU")

        logger.info("Personality model ready")

    except Exception as e:
        logger.error(f"Failed to load personality model: {e}")
        raise


def predict_personality(text: str) -> List[float]:
    """Predict Big Five personality traits from text.

    Args:
        text: Input text to analyze.

    Returns:
        List of 5 values (0-1 range): [extraversion, neuroticism, agreeableness, conscientiousness, openness]
    """
    global _model, _tokenizer, _device

    # Ensure model is loaded
    if _model is None:
        load_personality_model()

    try:
        # Tokenize input
        inputs = _tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = _model(**inputs)

        # Apply sigmoid to convert logits to probabilities (0-1 range)
        logits = outputs.logits.squeeze()
        predictions = torch.sigmoid(logits).cpu().numpy()

        # Return as list of floats
        return [float(predictions[i]) for i in range(5)]

    except Exception as e:
        logger.error(f"Personality prediction failed: {e}")
        return [0.5, 0.5, 0.5, 0.5, 0.5]  # Return neutral values on error


def get_personality_dict(text: str) -> Dict[str, float]:
    """Predict Big Five personality traits and return as dictionary.

    Args:
        text: Input text to analyze.

    Returns:
        Dictionary with trait names and scores (0-1 range).
    """
    predictions = predict_personality(text)
    trait_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    return {trait_names[i]: predictions[i] for i in range(5)}


def cleanup_personality_model():
    """Cleanup personality model and free GPU memory."""
    global _model, _tokenizer, _device

    if _model is not None:
        del _model
        _model = None

    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Personality model cleaned up")

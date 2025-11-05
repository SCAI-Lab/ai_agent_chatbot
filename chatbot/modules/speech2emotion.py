"""Speech emotion recognition powered by the parallel CNN + Transformer model from
https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch."""
import logging
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

# Model artefact packaged with the project (highest-accuracy model per README)
MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "cnn_transf_parallel_model.pt"
)

# Audio & feature extraction hyper-parameters matching the original training setup
TARGET_SAMPLE_RATE = 48_000
CLIP_DURATION_SECONDS = 3.0
CLIP_SAMPLES = int(TARGET_SAMPLE_RATE * CLIP_DURATION_SECONDS)
LIBROSA_LOAD_OFFSET = 0.5  # seconds
MEL_N_FFT = 1024
MEL_WIN_LENGTH = 512
MEL_HOP_LENGTH = 256
MEL_N_MELS = 128
MEL_FMAX = TARGET_SAMPLE_RATE / 2

# Emotion labels follow the integer encoding used during training
# Original model outputs 8 classes, we merge calm+neutral -> neutral
MODEL_EMOTIONS = [
    "surprise",
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
]

# Aligned 7-class labels matching text2emotion output
EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

_model: Optional[nn.Module] = None
_device: Optional[torch.device] = None


class ParallelModel(nn.Module):
    """Parallel 2D CNN and Transformer encoder architecture."""

    def __init__(self, num_emotions: int) -> None:
        super().__init__()
        self.conv2Dblock = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2, 4], stride=[2, 4])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=512,
            dropout=0.4,
            activation="relu",
        )
        self.transf_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.out_linear = nn.Linear(320, num_emotions)
        self.dropout_linear = nn.Dropout(p=0.0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_embedding = self.conv2Dblock(x)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1)

        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(2, 0, 1)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)

        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1)
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax


def _load_audio(audio_path: str) -> np.ndarray:
    """Load, trim/pad and normalise audio to the expected length."""
    try:
        audio, _ = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            duration=CLIP_DURATION_SECONDS,
            offset=LIBROSA_LOAD_OFFSET,
        )
        if audio.size == 0:
            # Fall back to reading from the start if offset truncates everything
            audio, _ = librosa.load(
                audio_path, sr=TARGET_SAMPLE_RATE, duration=CLIP_DURATION_SECONDS
            )
    except Exception as exc:
        logger.error("Failed to load audio with librosa: %s", exc)
        raise

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if audio.shape[0] >= CLIP_SAMPLES:
        return audio[:CLIP_SAMPLES]

    padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    padded[: audio.shape[0]] = audio
    return padded


def _extract_mel(audio: np.ndarray) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=MEL_N_FFT,
        win_length=MEL_WIN_LENGTH,
        hop_length=MEL_HOP_LENGTH,
        window="hamming",
        n_mels=MEL_N_MELS,
        fmax=MEL_FMAX,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db.astype(np.float32)


def _standardize(sample: np.ndarray) -> np.ndarray:
    """Standardise per-sample to mimic dataset-level scaling."""
    flat = sample.reshape(-1)
    mean = float(flat.mean())
    std = float(flat.std())
    if std < 1e-6:
        std = 1.0
    return (sample - mean) / std


def _prepare_input(audio_path: str) -> torch.Tensor:
    audio = _load_audio(audio_path)
    mel = _extract_mel(audio)
    mel = mel[np.newaxis, np.newaxis, :, :]
    normalised = _standardize(mel)
    tensor = torch.from_numpy(normalised)
    return tensor


def load_emotion_model() -> None:
    """Load the parallel CNN + Transformer emotion model."""
    global _model, _device

    if _model is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Emotion model weights not found at {MODEL_PATH}")

    logger.info("Loading CNN+Transformer emotion recognition model...")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = ParallelModel(num_emotions=len(MODEL_EMOTIONS))
    state_dict = torch.load(MODEL_PATH, map_location=_device)
    _model.load_state_dict(state_dict)
    _model = _model.to(_device)
    _model.eval()

    logger.info("Emotion model loaded on %s", _device)


def predict_emotion(audio_path: str, return_logits: bool = False) -> Dict[str, float]:
    """Predict emotion probabilities for the given audio clip.

    Args:
        audio_path: Path to audio file
        return_logits: If True, return logits instead of probabilities

    Returns:
        7-class logits or normalized probabilities aligned with text2emotion:
        anger, disgust, fear, happy, neutral, sad, surprise
        (merges original calm+neutral into neutral)
    """
    if _model is None or _device is None:
        load_emotion_model()

    try:
        inputs = _prepare_input(audio_path).to(_device)
        with torch.no_grad():
            logits, probabilities = _model(inputs)

        logits_np = logits.squeeze(0).cpu().numpy().astype(float)
        probs_np = probabilities.squeeze(0).cpu().numpy().astype(float)

        # Map 8-class model output to 7-class aligned labels
        # MODEL_EMOTIONS = [surprise, neutral, calm, happy, sad, angry, fear, disgust]
        # EMOTION_LABELS = [anger, disgust, fear, happy, neutral, sad, surprise]
        model_logits = {MODEL_EMOTIONS[i]: float(logits_np[i]) for i in range(len(MODEL_EMOTIONS))}
        model_probs = {MODEL_EMOTIONS[i]: float(probs_np[i]) for i in range(len(MODEL_EMOTIONS))}

        if return_logits:
            # For logits: merge by addition (log-space addition = multiplication in prob-space)
            # This is technically adding log-probabilities
            merged_neutral_logit = np.logaddexp(model_logits["neutral"], model_logits["calm"])

            aligned_logits = {
                "anger": model_logits["angry"],
                "disgust": model_logits["disgust"],
                "fear": model_logits["fear"],
                "happy": model_logits["happy"],
                "neutral": float(merged_neutral_logit),
                "sad": model_logits["sad"],
                "surprise": model_logits["surprise"]
            }
            return aligned_logits
        else:
            # For probabilities: merge by addition then renormalize
            merged_neutral = model_probs["neutral"] + model_probs["calm"]

            aligned_probs = {
                "anger": model_probs["angry"],
                "disgust": model_probs["disgust"],
                "fear": model_probs["fear"],
                "happy": model_probs["happy"],
                "neutral": merged_neutral,
                "sad": model_probs["sad"],
                "surprise": model_probs["surprise"]
            }

            # Renormalize after merging (to ensure sum=1.0)
            total = sum(aligned_probs.values())
            return {k: v / total for k, v in aligned_probs.items()}

    except Exception as exc:
        logger.error("Emotion prediction failed: %s", exc)
        if return_logits:
            # Return zero logits on error
            return {label: 0.0 for label in EMOTION_LABELS}
        else:
            uniform = 1.0 / len(EMOTION_LABELS)
            return {label: uniform for label in EMOTION_LABELS}

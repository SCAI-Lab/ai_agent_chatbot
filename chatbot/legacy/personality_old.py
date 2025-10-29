"""Big Five personality analysis module."""
import os
import pickle
from typing import List

from .config import PERSONALITY_MODELS_DIR, logger


def load_pickle_model(path: str):
    """Load a pickled scikit-learn model.

    Args:
        path: Path to the pickle file (absolute or relative to project root).

    Returns:
        Loaded model object.

    Raises:
        Exception: If model loading fails.
    """
    try:
        if not os.path.isabs(path):
            from .config import BASE_DIR
            abs_path = os.path.join(BASE_DIR, path)
        else:
            abs_path = path

        with open(abs_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model {path}: {e}")
        raise


# Load personality models and vectorizers
cEXT = load_pickle_model("data/models/cEXT.p")
cNEU = load_pickle_model("data/models/cNEU.p")
cAGR = load_pickle_model("data/models/cAGR.p")
cCON = load_pickle_model("data/models/cCON.p")
cOPN = load_pickle_model("data/models/cOPN.p")
vectorizer_31 = load_pickle_model("data/models/vectorizer_31.p")
vectorizer_30 = load_pickle_model("data/models/vectorizer_30.p")


def predict_personality(text: str) -> List[float]:
    """Predict Big Five personality traits from text.

    Args:
        text: Input text to analyze.

    Returns:
        List of 5 values: [extraversion, neuroticism, agreeableness, conscientiousness, openness]
    """
    try:
        sentences = text.split(". ")
        text_vector_31 = vectorizer_31.transform(sentences)
        text_vector_30 = vectorizer_30.transform(sentences)

        ext = cEXT.predict(text_vector_31)
        neu = cNEU.predict(text_vector_30)
        agr = cAGR.predict(text_vector_31)
        con = cCON.predict(text_vector_31)
        opn = cOPN.predict(text_vector_31)

        return [float(ext[0]), float(neu[0]), float(agr[0]), float(con[0]), float(opn[0])]

    except Exception as e:
        logger.error(f"Personality prediction failed: {e}")
        return [0.0, 0.0, 0.0, 0.0, 0.0]

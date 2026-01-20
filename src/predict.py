"""
Prediction utilities for the Income Prediction Project.
"""

import pandas as pd
import joblib
from pathlib import Path

try:
    from .config import config
    from .features.feature_engineering import create_features
except ImportError:
    from config import config
    from features.feature_engineering import create_features

from sklearn.pipeline import Pipeline

def load_model():
    """Load the trained model from disk."""
    model_path = config.MODELS_DIR / "income_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)

def load_preprocessor():
    """Load the preprocessor (if saved separately)."""
    preprocessor_path = config.MODELS_DIR / "preprocessor.pkl"
    if preprocessor_path.exists():
        return joblib.load(preprocessor_path)
    else:
        # Assume preprocessing is done in the model pipeline
        return None

def predict_income(features):
    """
    Predict income category for given features.

    Args:
        features (dict or pd.DataFrame): Input features for prediction.

    Returns:
        str: Predicted income category ('<=50K' or '>50K').
    """
    model = load_model()
    preprocessor = load_preprocessor()

    if isinstance(features, dict):
        features = pd.DataFrame([features])

    # Ensure we apply the same feature engineering used during training
    try:
        features = create_features(features)
    except Exception:
        # If feature creation fails for any reason, proceed without it
        pass

    # If the model is a pipeline that already includes a preprocessor step,
    # do not apply the separately-saved preprocessor again (would double-transform).
    if isinstance(model, Pipeline) and 'preprocessor' in getattr(model, 'named_steps', {}):
        features_processed = features
    elif preprocessor is not None:
        features_processed = preprocessor.transform(features)
    else:
        features_processed = features

    prediction = model.predict(features_processed)
    return prediction[0]  # Assuming single prediction

def predict_proba_income(features):
    """
    Predict income probabilities for given features.

    Args:
        features (dict or pd.DataFrame): Input features for prediction.

    Returns:
        np.array: Probabilities for each class.
    """
    model = load_model()
    preprocessor = load_preprocessor()

    if isinstance(features, dict):
        features = pd.DataFrame([features])

    # Apply same feature engineering as during training
    try:
        features = create_features(features)
    except Exception:
        pass

    if isinstance(model, Pipeline) and 'preprocessor' in getattr(model, 'named_steps', {}):
        features_processed = features
    elif preprocessor is not None:
        features_processed = preprocessor.transform(features)
    else:
        features_processed = features

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_processed)
        return probabilities[0]
    else:
        raise AttributeError("Model does not support probability prediction.")
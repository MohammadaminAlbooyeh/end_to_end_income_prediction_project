"""
Prediction utilities for the Income Prediction Project.
"""

import pandas as pd
import joblib
from pathlib import Path
from .config import config

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

    if preprocessor:
        features_processed = preprocessor.transform(features)
    else:
        # Assume features are already processed or model handles it
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

    if preprocessor:
        features_processed = preprocessor.transform(features)
    else:
        features_processed = features

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_processed)
        return probabilities[0]
    else:
        raise AttributeError("Model does not support probability prediction.")
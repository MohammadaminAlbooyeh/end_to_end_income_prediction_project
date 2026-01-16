"""
Training script for the Income Prediction Project.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .config import config
from .data_loader import load_raw_data
from .utils.preprocessing import preprocess_pipeline
from .features.feature_engineering import create_features

def preprocess_data(df):
    """Wrapper preprocessing used by training and tests.

    Uses helpers in `src.utils.preprocessing` and applies lightweight feature engineering.
    """
    # deterministic preprocessing
    df_proc = preprocess_pipeline(df)

    # add derived features (kept additive so original columns remain)
    df_proc = create_features(df_proc)

    return df_proc

def create_preprocessor():
    """Create a preprocessor for categorical and numerical features."""
    categorical_transformer = LabelEncoder()  # For simplicity, but actually need OneHot or Ordinal
    numerical_transformer = StandardScaler()

    # For pipeline, better to use ColumnTransformer with OneHotEncoder
    from sklearn.preprocessing import OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, config.NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), config.CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def train_model():
    """Train the income prediction model."""
    # Load and preprocess data
    df = load_raw_data()
    df = preprocess_data(df)

    # Split features and target
    X = df.drop(config.TARGET, axis=1)
    y = df[config.TARGET]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Create pipeline
    preprocessor = create_preprocessor()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE
        ))
    ])

    # Train
    model.fit(X_train, y_train)

    # Save model
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "income_model.pkl")

    # Save preprocessor separately if needed
    joblib.dump(preprocessor, config.MODELS_DIR / "preprocessor.pkl")

    print("Model trained and saved.")

    # Optional: evaluate on test set
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    return model

if __name__ == "__main__":
    train_model()
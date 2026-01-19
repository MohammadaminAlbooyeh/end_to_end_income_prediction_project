"""
Configuration settings for the Income Prediction Project.
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the income prediction project."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    DOCS_DIR = PROJECT_ROOT / "docs"

    # Data settings
    RAW_DATA_FILE = "adult.csv"  # Assuming Adult Census dataset
    PROCESSED_DATA_FILE = "processed_data.pkl"
    TRAIN_DATA_FILE = "train_data.pkl"
    TEST_DATA_FILE = "test_data.pkl"

    # Model hyperparameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 100  # For Random Forest
    MAX_DEPTH = 10

    # Feature engineering
    CATEGORICAL_FEATURES = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    NUMERICAL_FEATURES = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                          'capital-loss', 'hours-per-week']
    TARGET = 'income'

    # Training settings
    CV_FOLDS = 5
    SCORING = 'accuracy'

# Global config instance
config = Config()
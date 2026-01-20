# API Documentation

## Overview

This document provides API documentation for the Income Prediction project modules.

## Module: src.config

Configuration settings for the project.

### Config Class

```python
from src.config import config
```

**Attributes:**
- `PROJECT_ROOT`: Path to project root
- `DATA_DIR`: Path to data directory
- `MODELS_DIR`: Path to models directory
- `RAW_DATA_FILE`: Filename for raw dataset
- `TARGET`: Target column name ('income')
- `CATEGORICAL_FEATURES`: List of categorical feature names
- `NUMERICAL_FEATURES`: List of numerical feature names
- `RANDOM_STATE`: Random seed (42)
- `TEST_SIZE`: Train/test split ratio (0.2)
- `N_ESTIMATORS`: Number of trees in Random Forest (100)
- `MAX_DEPTH`: Maximum depth of trees (10)

## Module: src.data_loader

Data loading utilities.

### Functions

#### `download_adult_dataset()`

Downloads the Adult dataset from UCI repository if not present.

#### `load_raw_data()`

Loads the raw Adult dataset into a pandas DataFrame.

**Returns:** pandas.DataFrame

#### `load_processed_data()`

Loads processed data if available, otherwise loads and processes raw data.

**Returns:** pandas.DataFrame

## Module: src.train

Model training functionality.

### Functions

#### `preprocess_data(df)`

Applies preprocessing pipeline to raw data.

**Parameters:**
- `df` (pandas.DataFrame): Raw data

**Returns:** pandas.DataFrame - Processed data

#### `create_preprocessor()`

Creates the sklearn ColumnTransformer for preprocessing.

**Returns:** sklearn.compose.ColumnTransformer

#### `train_model()`

Trains the income prediction model and saves it.

**Returns:** sklearn.pipeline.Pipeline - Trained model

## Module: src.predict

Prediction functionality.

### Functions

#### `load_model()`

Loads the trained model from disk.

**Returns:** sklearn.pipeline.Pipeline

#### `load_preprocessor()`

Loads the preprocessor (if saved separately).

**Returns:** sklearn.compose.ColumnTransformer or None

#### `predict_income(features)`

Predicts income category for given features.

**Parameters:**
- `features` (dict or pandas.DataFrame): Input features

**Returns:** str - Predicted income category ('<=50K' or '>50K')

#### `predict_proba_income(features)`

Predicts income probabilities for given features.

**Parameters:**
- `features` (dict or pandas.DataFrame): Input features

**Returns:** numpy.ndarray - Prediction probabilities [P(<=50K), P(>50K)]

## Module: src.utils.preprocessing

Preprocessing utilities.

### Functions

#### `fill_missing_values(df)`

Fills missing values: mode for categorical, median for numerical.

**Parameters:**
- `df` (pandas.DataFrame): Input data

**Returns:** pandas.DataFrame

#### `encode_target(df, target_col=None)`

Encodes the income target to 0/1.

**Parameters:**
- `df` (pandas.DataFrame): Input data
- `target_col` (str, optional): Target column name

**Returns:** pandas.DataFrame

#### `preprocess_pipeline(df, save=False)`

Runs full preprocessing pipeline.

**Parameters:**
- `df` (pandas.DataFrame): Raw data
- `save` (bool): Whether to save processed data

**Returns:** pandas.DataFrame

## Module: src.features.feature_engineering

Feature engineering utilities.

### Functions

#### `create_features(df)`

Adds derived features to the dataset.

**Adds:**
- `capital_gain_binary`: 1 if capital-gain > 0, else 0
- `age_bucket`: Categorical age buckets

**Parameters:**
- `df` (pandas.DataFrame): Input data

**Returns:** pandas.DataFrame

## Error Handling

All prediction functions include error handling for:
- Missing model files
- Invalid input data
- Preprocessing failures

## Examples

See the `notebooks/` directory for usage examples and the main `README.md` for quick start guides.
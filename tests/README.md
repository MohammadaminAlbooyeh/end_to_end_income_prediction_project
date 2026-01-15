# Tests Directory

This directory contains the test suite for the Income Prediction Project.

## Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_config.py` - Tests for configuration module
- `test_data_loader.py` - Tests for data loading functionality
- `test_train.py` - Tests for model training
- `test_predict.py` - Tests for prediction functionality
- `test_integration.py` - Integration tests for the complete pipeline
- `pytest.ini` - Pytest configuration file

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_config.py
```

### Run specific test class
```bash
pytest tests/test_train.py::TestPreprocessData
```

### Run specific test function
```bash
pytest tests/test_train.py::TestPreprocessData::test_preprocess_data_returns_dataframe
```

### Run with coverage
```bash
pytest --cov=src --cov-report=html
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

### Run with verbose output
```bash
pytest -v
```

### Run with print statements visible
```bash
pytest -s
```

## Test Categories

### Unit Tests
- `test_config.py` - Configuration validation
- `test_data_loader.py` - Data loading functions
- `test_train.py` - Training functions (preprocessing, model creation)
- `test_predict.py` - Prediction functions

### Integration Tests
- `test_integration.py` - End-to-end pipeline tests

## Fixtures

Common fixtures available in `conftest.py`:

- `sample_adult_data` - Sample Adult Census dataset
- `sample_features_dict` - Sample feature dictionary for prediction
- `temp_data_dir` - Temporary directory for data files
- `temp_models_dir` - Temporary directory for model files
- `mock_config` - Mocked configuration object
- `sample_preprocessed_data` - Preprocessed sample data

## Writing New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py`
2. Organize tests into classes: `TestClassName`
3. Use descriptive test names: `test_function_does_something`
4. Use fixtures for common setup
5. Add appropriate markers for test categorization
6. Include docstrings explaining what each test validates

## Dependencies

Required packages for testing:
- pytest
- pytest-cov (optional, for coverage reports)
- pandas
- numpy
- scikit-learn
- joblib

Install with:
```bash
pip install pytest pytest-cov
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Ensure all tests pass before merging code.

## Coverage Goals

Aim for:
- Unit tests: >80% code coverage
- Integration tests: Cover all major workflows
- Edge cases: Test error handling and boundary conditions

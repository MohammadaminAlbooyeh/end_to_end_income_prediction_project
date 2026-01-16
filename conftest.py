"""Compatibility shim so tests that import `conftest` directly work.

This re-exports fixtures defined in `tests/conftest.py` so `from conftest import ...`
imports succeed when running tests under different import contexts.
"""
import tests.conftest as _tc

# Some tests import `conftest` directly and call fixtures as functions.
# Expose plain callables by using the underlying wrapped functions when available.
def _unwrap(func):
    return getattr(func, '__wrapped__', func)

sample_adult_data = _unwrap(_tc.sample_adult_data)
sample_features_dict = _unwrap(_tc.sample_features_dict)
temp_data_dir = _unwrap(_tc.temp_data_dir)
temp_models_dir = _unwrap(_tc.temp_models_dir)
mock_config = _unwrap(_tc.mock_config)
sample_preprocessed_data = _unwrap(_tc.sample_preprocessed_data)

__all__ = [
    'sample_adult_data',
    'sample_features_dict',
    'temp_data_dir',
    'temp_models_dir',
    'mock_config',
    'sample_preprocessed_data',
]

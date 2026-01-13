"""
Income Prediction Package

This package provides tools for end-to-end income prediction,
including data loading, feature engineering, model training, and prediction.
"""

from . import config
from . import data_loader
from . import predict
from . import train
from . import features
from . import models
from . import utils

__version__ = "0.1.0"
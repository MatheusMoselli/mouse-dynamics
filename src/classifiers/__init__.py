"""
Centralizes the logic for fitting the data and getting the expected results.
"""
from .base_classifier import BaseClassifier
from .random_forest_classifier import RandomForestClassifier

__all__ = [
    "BaseClassifier",
    "RandomForestClassifier",
]

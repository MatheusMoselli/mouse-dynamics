"""
Centralizes the logic for reading datasets and standardizing the resulting dataframes.
"""
from enum import Enum
from .UserDataDto import UserDataDto
from .ExtractionData import ExtractionData

class EnumTypeOfSession(Enum):
    TRAINING = "training"
    TESTING = "testing"

__all__ = [
    "UserDataDto",
    "ExtractionData",
    "EnumTypeOfSession"
]
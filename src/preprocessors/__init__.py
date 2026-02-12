"""
Centralizes the logic for feature extraction and statistical analysis upon these features.
"""
from enum import Enum
from .base_preprocessor import BasePreprocessor
from .minecraft_preprocessor import  MinecraftPreprocessor
from .khan_preprocessor import KhanPreprocessor

class EnumPreprocessors(Enum):
    MINECRAFT = "minecraft",
    KHAN = "khan",

def load_preprocessor(preprocessor_name: EnumPreprocessors, is_debug: bool) -> BasePreprocessor:
    """
    Factory function to preprocess the dataframes by users.

    :param preprocessor_name: Name of the preprocessor.
    :param is_debug: Is the preprocessor being run in debug mode.
    :return: Dictionary mapping user_id to DataFrame
    """
    preprocessors = {
        EnumPreprocessors.MINECRAFT: MinecraftPreprocessor,
        EnumPreprocessors.KHAN: KhanPreprocessor,
    }

    if preprocessor_name in preprocessors:
        preprocessor = preprocessors[preprocessor_name](is_debug)
        return preprocessor
    else:
        raise ValueError(f"Unknown preprocessor: {preprocessor_name}")

__all__ = [
    "BasePreprocessor",
    "EnumPreprocessors",
    "load_preprocessor"
]
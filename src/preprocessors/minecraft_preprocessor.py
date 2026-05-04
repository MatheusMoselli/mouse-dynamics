"""
Preprocessing following `Continuous Authentication Using Mouse Movements,
Machine Learning, and Minecraft`.
"""
from src.preprocessors import BasePreprocessor


class MinecraftPreprocessor(BasePreprocessor):
    """
    Preprocessor for the Minecraft dataset.

    Inherits the full pipeline from BasePreprocessor. The window size of 10
    matches the granularity used in the reference article.
    """
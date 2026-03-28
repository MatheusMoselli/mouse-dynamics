"""
Preprocessing following `Mouse Dynamics Behavioral Biometrics: A Survey`.
"""
from src.preprocessors import BasePreprocessor


class KhanPreprocessor(BasePreprocessor):
    """
    Preprocessor for the Khan / Balabit dataset.

    Inherits the full pipeline from BasePreprocessor. The window size of 10
    matches the granularity used in the reference article.
    """

    WINDOW_SIZE: int = 10
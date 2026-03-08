"""
Base classifier for better abstraction and dependency injection
"""
from abc import ABC, abstractmethod
from src.dto import ExtractionData

class BaseClassifier(ABC):
    """
    Abstraction for all classifiers.
    """
    def __init__(self, is_debug: bool = False):
        """
        Class initialization.
        :param is_debug: Is the classifier being run in debug mode.
        """
        self.is_debug = is_debug

    @abstractmethod
    def fit (self, extraction_data: ExtractionData):
        """
        Fit the user`s datas into the desired classifiers, printing the results in the console.

        :param extraction_data: The user`s dataframes.
        """
        pass

"""
Base classifier for better abstraction and dependency injection
"""
from abc import ABC, abstractmethod
from typing import Dict
from pandas import DataFrame

class BaseClassifier(ABC):
    """
    Abstraction for all classifiers.
    """

    @abstractmethod
    def fit (self, dataframes_by_users: Dict[str, DataFrame]):
        """
        Fit the user`s datas into the desired classifiers, printing the results in the console.

        :param dataframes_by_users: The user`s dataframes.
        """
        pass

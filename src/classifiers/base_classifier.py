"""
Base classifier for better abstraction and dependency injection
"""
from src.dto import ExtractionData, UserDataDto, EnumTypeOfSession
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

_DROP_COLS = ["authentic"]

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

    @staticmethod
    def _prepare_user_data(
            user: UserDataDto
    ) -> Optional[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Validate, merge, and split one user's sessions into model-ready arrays.

        Returns None (and logs the reason) when the user should be skipped.
        Otherwise, returns (x_train, y_train, x_test, y_test).

        Concrete classifiers call this at the top of their per-user loop so
        that validation and feature/label splitting are not duplicated across
        every classifier implementation.

        :param user: UserDataDto after preprocessing and splitting
        :return: (x_train, y_train, x_test, y_test) or None
        """
        if not user.is_user_valid():
            return None

        train_df = user.merged_sessions(EnumTypeOfSession.TRAINING).dropna()
        test_df = user.merged_sessions(EnumTypeOfSession.TESTING).dropna()

        if train_df.empty or test_df.empty:
            return None

        x_train = train_df.drop(columns=_DROP_COLS)
        y_train = train_df["authentic"]

        x_test = test_df.drop(columns=_DROP_COLS)
        y_test = test_df["authentic"]

        return x_train, y_train, x_test, y_test
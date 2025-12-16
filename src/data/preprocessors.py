"""
Data preprocessing utilities for mouse dynamics.
Handles invalid data and movement segmentation.
"""
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BasePreprocessor(ABC):
    """Abstract base class for preprocessors."""
    @abstractmethod
    def preprocess(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        #TODO: Add typing to the config dictionary
        """
        Preprocess the given dataframe.

        :param data: Initial dataframe.
        :param config: Configuration dictionary.
        :return: Preprocessed dataframe.
        """
        pass


class PreprocessorByMouseClick(BasePreprocessor):
    """Preprocessor that joins lines between mouse clicks"""

    #TODO: Implement
    def preprocess(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Preprocess the given dataframe.

        :param df: Raw trajectory DataFrame

        :return: Preprocessed DataFrame
        """
        df = df.copy()

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Remove invalid rows
        df = df.dropna(subset=['x', 'y', 'timestamp'])

        # Ensure timestamps are relative to start
        if len(df) > 0 and df['timestamp'][0] != 0:
            df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]

        return df.reset_index(drop=True)


class PreprocessorByMouseAction(BasePreprocessor):
    """Preprocessor that joins lines between mouse actions (drag and drop, click, among others)."""

    #TODO: Implement
    def preprocess(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Preprocess the given dataframe.

        :param df: Raw trajectory DataFrame

        :return: Preprocessed DataFrame
        """
        df = df.copy()

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Remove invalid rows
        df = df.dropna(subset=['x', 'y', 'timestamp'])

        # Ensure timestamps are relative to start
        if len(df) > 0 and df['timestamp'][0] != 0:
            df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]

        return df.reset_index(drop=True)

class PreprocessorByDefinedAmountOfLines(BasePreprocessor):
    """Preprocessor that joins lines in groups of a defined amount of lines."""

    #TODO: Implement
    def preprocess(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Preprocess the given dataframe.

        :param df: Raw trajectory DataFrame

        :return: Preprocessed DataFrame
        """
        df = df.copy()

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Remove invalid rows
        df = df.dropna(subset=['x', 'y', 'timestamp'])

        # Ensure timestamps are relative to start
        if len(df) > 0 and df['timestamp'][0] != 0:
            df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]

        return df.reset_index(drop=True)

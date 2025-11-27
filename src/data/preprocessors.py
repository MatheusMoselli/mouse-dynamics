"""
Data preprocessing utilities for mouse dynamics.
Handles cleaning, normalization, and trajectory segmentation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class MouseDynamicsPreprocessor:
    """Main preprocessing pipeline for mouse dynamics data."""

    def __init__(self,
                 remove_duplicates: bool = True):
                 # remove_stationary: bool = True):
        """
        Initialize the preprocessor.

        Args:
            remove_duplicates: Remove consecutive duplicate points
            remove_stationary: Remove points where mouse doesn't move
        """
        self.remove_duplicates = remove_duplicates
        # self.remove_stationary = remove_stationary

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline.

        Args:
            df: Raw trajectory DataFrame

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Remove invalid rows
        df = df.dropna(subset=['x', 'y', 'timestamp'])

        if self.remove_duplicates:
            df = self._remove_duplicate_points(df)

        # if self.remove_stationary:
        #     df = self._remove_stationary_points(df)

        # Ensure timestamps are relative to start
        if len(df) > 0:
            df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]

        return df.reset_index(drop=True)

    def _remove_duplicate_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove consecutive duplicate points."""
        # Keep first occurrence of duplicates
        mask = (df['x'].diff() != 0) | (df['y'].diff() != 0)
        mask.iloc[0] = True  # Keep first point
        return df[mask]

    # def _remove_stationary_points(self, df: pd.DataFrame,
    #                               threshold: float = 1.0) -> pd.DataFrame:
    #     """Remove points where mouse movement is below threshold."""
    #     if len(df) < 2:
    #         return df
    #
    #     # Calculate distance moved
    #     dx = df['x'].diff()
    #     dy = df['y'].diff()
    #     distance = np.sqrt(dx ** 2 + dy ** 2)
    #
    #     # Keep points above threshold
    #     mask = distance >= threshold
    #     mask.iloc[0] = True  # Keep first point
    #
    #     return df[mask]
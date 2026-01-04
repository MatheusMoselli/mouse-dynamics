"""
Generic preprocessor for feature extraction and statistical analysis.
"""
from typing import Dict
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BasePreprocessor(ABC):
    """
    Extract behavioral biometric features from mouse trajectory data.

    All features are computed using vectorized NumPy operations for maximum performance.\

    The extractor can process trajectories in sequential windows, extracting features
    from each window to create a sequence of feature vectors.
    """
    @abstractmethod
    def preprocess(self,
                   dataframes_by_users: Dict[str, pd.DataFrame],
                   curvature_threshold: float = 0.0005,
                   is_debug: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all the dataframes by user

        :param is_debug: If true, will save a parquet file for each user, with its features extracted
        :param dataframes_by_users: The users standardized dataframes
        :param curvature_threshold: Threshold for detecting critical curvature points (THc)

        :return: Dataframe containing all extracted features with descriptive names
        """
        pass

    @abstractmethod
    def _extract_general_features_from_df(self, dataframe: pd.DataFrame)-> pd.DataFrame:
        """
        Extract all movement features from a trajectory.

        :param dataframe: DataFrame to extract features from
        :return: Dataframe containing all extracted features with descriptive names
        """
        pass

    @abstractmethod
    def _extract_statistical_info_from_df(self, general_features_df: pd.DataFrame)-> pd.DataFrame:
        """
        Use the features extracted (such as velocity, acceleration, angle) and group the lines following
        the desired pattern to calculate statistical information (mean, standard deviation, min, max).

        :param general_features_df: The dataframe with general features extracted
        :return: A dataframe with the statistical features extracted
        """
        pass

    @abstractmethod
    def _extract_temporal_features(self, time_arr: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features.

        These features describe timing characteristics of the mouse movement.

        :param time_arr: The timestamp array
        :return: A dictionary with temporal related features
        """
        pass

    @abstractmethod
    def _extract_kinematic_features(self, x_axis_arr: np.ndarray, y_axis_arr: np.ndarray, time_arr: np.ndarray) -> Dict[str, float]:
        """
        Extract kinematic features (velocity, acceleration, jerk).

        These features describe the motion characteristics of the mouse cursor.

        :param x_axis_arr: the X coordinate array
        :param y_axis_arr: the Y coordinate array
        :param time_arr: the timestamp array
        :return: A dictionary with the kinematic related features
        """
        pass

    @abstractmethod
    def _extract_curvature_features(self, x_axis_arr: np.ndarray, y_axis_arr: np.ndarray, time_arr: np.ndarray) -> Dict[str, float]:
        """
        Extract curvature-related features.

        These features describe how the trajectory curves and changes direction.

        :param x_axis_arr: the X coordinate array
        :param y_axis_arr: the Y coordinate array
        :param time_arr: the timestamp array
        :return: A dictionary with the curvature related features
        """
        pass

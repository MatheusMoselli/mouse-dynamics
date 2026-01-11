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

    def __init__(self, is_debug: bool = False):
        """
        Class initialization.
        :param is_debug: Is the classifier being run in debug mode.
        """
        self.is_debug = is_debug

    @abstractmethod
    def preprocess(self,
                   dataframes_by_users: Dict[str, pd.DataFrame],
                   curvature_threshold: float = 0.0005) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all the dataframes by user
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

    def _extract_temporal_features(self, time_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract temporal features.

        These features describe timing characteristics of the mouse movement.

        :param time_arr: The timestamp array
        :return: A dictionary with temporal related features
        """
        temporal_features = {
            'elapsed_time': np.concatenate(([0], np.diff(time_arr)))
        }

        return temporal_features

    def _extract_kinematic_features(self, x_axis_arr: np.ndarray, y_axis_arr: np.ndarray, time_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract kinematic features (velocity, acceleration, jerk).

        These features describe the motion characteristics of the mouse cursor.

        :param x_axis_arr: the X coordinate array
        :param y_axis_arr: the Y coordinate array
        :param time_arr: the timestamp array
        :return: A dictionary with the kinematic related features
        """
        # TODO: check literature to see if I should use the absolute values of x - x-1 and y - y-1 (std is too big)
        motion_features = {}


        # TODO: Pass the thing that make the first lines of the dataframe as 0 to the minecraft_preprocessor if necessary
        # Calculate time differences
        time_differences_arr = np.concatenate(([0], np.diff(time_arr)))

        # Calculate spatial differences
        x_differences_arr = np.concatenate(([0], np.diff(x_axis_arr)))
        y_differences_arr = np.concatenate(([0], np.diff(y_axis_arr)))

        # Speed magnitude {sqrt(x_differences_arr*2 + y_differences_arr*2) / time_differences_arr}
        speed_arr = np.zeros_like(time_differences_arr)
        np.divide(np.sqrt(x_differences_arr ** 2 + y_differences_arr ** 2), time_differences_arr, out=speed_arr,
                  where=time_differences_arr != 0)
        motion_features['speed'] = speed_arr

        # Horizontal speed (x-axis) {x_differences_arr / time_differences_arr}
        x_speed_arr = np.zeros_like(time_differences_arr)
        # TODO: check my own audio recording in my personal whatsapp group. Talk to professor about it (duplicates interfering in timestamp difference)
        np.divide(x_differences_arr, time_differences_arr, out=x_speed_arr, where=time_differences_arr != 0)
        motion_features['speed_x'] = x_speed_arr

        # Vertical speed (y-axis) {y_differences_arr / time_differences_arr}
        y_speed_arr = np.zeros_like(time_differences_arr)
        np.divide(y_differences_arr, time_differences_arr, out=y_speed_arr, where=time_differences_arr != 0)
        motion_features['speed_y'] = y_speed_arr

        # Acceleration magnitude {speed_difference_arr / time_differences_arr}
        # TODO: talk to professor: in the original dataset, they calculate the acceleration wrong: dt / dv
        speed_difference_arr = np.concatenate(([0], np.diff(speed_arr)))

        acceleration_arr = np.zeros_like(time_differences_arr)
        np.divide(speed_difference_arr, time_differences_arr, out=acceleration_arr, where=time_differences_arr != 0)
        motion_features['acceleration'] = acceleration_arr

        # Horizontal Acceleration  (x-axis) {x_speed_difference_arr / time_differences_arr}
        x_speed_difference_arr = np.concatenate(([0], np.diff(x_speed_arr)))

        x_acceleration = np.zeros_like(time_differences_arr)
        np.divide(x_speed_difference_arr, time_differences_arr, out=x_acceleration, where=time_differences_arr != 0)
        motion_features['acceleration_x'] = x_acceleration

        # Vertical Acceleration  (y-axis) {y_speed_difference_arr / time_differences_arr}
        y_speed_difference_arr = np.concatenate(([0], np.diff(y_speed_arr)))

        y_acceleration = np.zeros_like(time_differences_arr)
        np.divide(y_speed_difference_arr, time_differences_arr, out=y_acceleration, where=time_differences_arr != 0)
        motion_features['acceleration_y'] = y_acceleration

        y_acceleration_difference_arr = np.concatenate(([0], np.diff(y_acceleration)))
        x_acceleration_difference_arr = np.concatenate(([0], np.diff(x_acceleration)))

        acceleration_difference_arr = np.concatenate(([0], np.diff(acceleration_arr)))

        jerk = np.zeros_like(time_differences_arr)
        np.divide(acceleration_difference_arr, time_differences_arr, out=jerk, where=time_differences_arr != 0)
        motion_features['jerk'] = jerk

        curvatures = np.zeros_like(time_differences_arr)
        curvature_upper_part = x_speed_difference_arr * y_acceleration_difference_arr - y_speed_difference_arr * x_acceleration_difference_arr
        curvature_lower_part = np.power(x_speed_difference_arr ** 2 + y_speed_difference_arr ** 2, 3 / 2)

        np.divide(curvature_upper_part, curvature_lower_part, out=curvatures, where=curvature_lower_part != 0)
        motion_features['curvature'] = curvatures

        return motion_features

    def _extract_curvature_features(self, x_axis_arr: np.ndarray, y_axis_arr: np.ndarray, time_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract curvature-related features.

        These features describe how the trajectory curves and changes direction.

        :param x_axis_arr: the X coordinate array
        :param y_axis_arr: the Y coordinate array
        :param time_arr: the timestamp array
        :return: A dictionary with the curvature related features
        """
        curvature_features = {}
        # Calculate time differences

        # Calculate spatial differences
        x_differences_arr = np.concatenate(([0], np.diff(x_axis_arr)))
        y_differences_arr = np.concatenate(([0], np.diff(y_axis_arr)))

        # Movement angle - Tangent angle relative to x-axis
        slopes = np.zeros_like(x_differences_arr)
        np.divide(y_differences_arr, x_differences_arr, out=slopes, where=x_differences_arr != 0)

        radian_angles = np.arctan(slopes)
        curvature_features['angle_radians'] = radian_angles
        curvature_features['angle_degrees'] = np.degrees(radian_angles)

        # Calculate time differences (avoid division by zero)
        # dt = np.diff(t)
        # dt = np.where(dt == 0, 1e-10, dt)
        #
        # if len(x) < 3:
        #     # Not enough points for curvature calculation
        #     features['curvatura_media'] = 0.0
        #     features['velocidade_curvatura_media'] = 0.0
        #     features['taxa_curvatura_media'] = 0.0
        #     features['angulo_movimento_medio'] = 0.0
        #     features['raio_curvatura_medio'] = 0.0
        #     features['velocidade_angular_media'] = 0.0
        #     features['angulos_totais_energia_curvatura'] = 0.0
        #     return features
        #
        # # First derivatives (velocity components)
        # dx = np.diff(x)
        # dy = np.diff(y)
        #
        # # Second derivatives (acceleration components)
        # if len(x) > 2:
        #     ddx = np.diff(dx) / dt[1:]
        #     ddy = np.diff(dy) / dt[1:]
        #
        #     # Velocity components at points where we have acceleration
        #     vx = dx[1:] / dt[1:]
        #     vy = dy[1:] / dt[1:]
        #
        #     # Curvature calculation: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        #     numerator = np.abs(vx * ddy - vy * ddx)
        #     denominator = (vx**2 + vy**2)**1.5
        #     denominator = np.where(denominator == 0, 1e-10, denominator)
        #
        #     curvature = numerator / denominator
        #
        #     # === CURVATURE STATISTICS ===
        #
        #     features['curvatura_media'] = np.mean(curvature)
        #     features['curvatura_std'] = np.std(curvature)
        #     features['curvatura_max'] = np.max(curvature)
        #
        #     # Raio de Curvatura médio (radius of curvature = 1/κ)
        #     radii = 1.0 / (curvature + 1e-10)
        #     # Filter out extreme values for stability
        #     features['raio_curvatura_medio'] = np.mean(radii[radii < 1e8])
        #
        #     # Velocidade de Curvatura (Vcurve) - Rate of change of curvature magnitude
        #     if len(curvature) > 1:
        #         vcurve = np.sqrt(curvature[:-1]**2 + curvature[1:]**2)
        #         features['velocidade_curvatura_media'] = np.mean(vcurve)
        #
        #     # Taxa de Curvatura - Time derivative of curvature
        #     if len(curvature) > 1:
        #         dcurvature = np.diff(curvature) / dt[2:]
        #         features['taxa_curvatura_media'] = np.mean(np.abs(dcurvature))
        #
        #     # Ângulos Totais / Energia de Curvatura - Sum of absolute curvatures
        #     features['angulos_totais_energia_curvatura'] = np.sum(np.abs(curvature))
        #
        # # === ANGULAR FEATURES ===
        #
        # # Velocidade Angular (ωi) - Rate of change of angle
        # if len(angles) > 1:
        #     angular_velocity = np.diff(angles) / dt[1:]
        #     # Handle angle wrapping (convert to [-π, π] range)
        #     angular_velocity = np.arctan2(np.sin(angular_velocity), np.cos(angular_velocity))
        #     features['velocidade_angular_media'] = np.mean(np.abs(angular_velocity))
        #     features['velocidade_angular_std'] = np.std(angular_velocity)
        #
        # # Ângulo (γ) - Lei dos Cossenos - Angle between consecutive segments
        # if len(x) > 2:
        #     angles_cosine = self._calculate_cosine_angles(x, y)
        #     features['angulo_lei_cossenos_medio'] = np.mean(angles_cosine)
        #     features['angulo_lei_cossenos_std'] = np.std(angles_cosine)

        return curvature_features


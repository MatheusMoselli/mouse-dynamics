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

    # Basic Axis
    _diff_x_axis_arr = np.array([])
    _diff_y_axis_arr = np.array([])
    _diff_time_arr = np.array([])
    _traveled_distance = np.array([])
    _curve_length = np.array([])

    # Speed
    _speed = np.array([])
    _horizontal_speed = np.array([])
    _vertical_speed = np.array([])

    # Acceleration
    _acceleration = np.array([])
    _horizontal_acceleration = np.array([])
    _vertical_acceleration = np.array([])

    _extracted_features = { }
    __features_dataframe: pd.DataFrame | None = None

    def __init__(self, is_debug: bool = False):
        """
        Class initialization.
        :param is_debug: Is the classifier being run in debug mode.
        """
        self.is_debug = is_debug

    @property
    def features_dataframe(self):
        if self.__features_dataframe is None:
            self.__features_dataframe = pd.DataFrame(self._extracted_features)

        return self.__features_dataframe

    @abstractmethod
    def preprocess(self, dataframes_by_users: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all the dataframes by user
        :param dataframes_by_users: The users standardized dataframes
        :param curvature_threshold: Threshold for detecting critical curvature points (THc)

        :return: Dataframe containing all extracted features with descriptive names
        """
        pass

    @abstractmethod
    def _extract_general_features_from_df(self, dataframe: pd.DataFrame):
        """
        Extract all movement features from a trajectory.

        :param dataframe: DataFrame to extract features from
        :return: Dataframe containing all extracted features with descriptive names
        """
        pass

    @abstractmethod
    def _extract_statistical_info_from_features_df(self)-> pd.DataFrame:
        """
        Use the features extracted (such as velocity, acceleration, angle) and group the lines following
        the desired pattern to calculate statistical information (mean, standard deviation, min, max).

        :param general_features_df: The dataframe with general features extracted
        :return: A dataframe with the statistical features extracted
        """
        pass

    def _calculate_basic_axis_features(self, x_axis_arr: np.ndarray, y_axis_arr: np.ndarray, time_arr: np.ndarray):
        self.__features_dataframe = None
        self._extracted_features = { "x": x_axis_arr, "y": y_axis_arr, "timestamp": time_arr }
        self._diff_x_axis_arr = np.concatenate(([0], np.diff(x_axis_arr)))
        self._diff_y_axis_arr = np.concatenate(([0], np.diff(y_axis_arr)))
        self._diff_time_arr = np.concatenate(([0], np.diff(time_arr)))

        window_size = 10

        # 1. Traveled Distance (Di)
        self._traveled_distance = np.sqrt(
            np.power(self._diff_x_axis_arr, 2) + np.power(self._diff_y_axis_arr, 2)
        )

        self._extracted_features["traveled_distance"] = self._traveled_distance

        # 2. Curve length/Real Dist. (Sn)
        self.curve_length = np.cumsum(self._traveled_distance)
        self._extracted_features["curve_length"] = self.curve_length

        # 3. Elapsed Time/Mouse Digraph
        self._extracted_features["elapsed_time"] = self._diff_time_arr

        # 4. Movement Offset
        movement_offset =  self.curve_length - self._traveled_distance
        self._extracted_features["movement_offset"] = movement_offset

        # 5. Deviation Distance
        #TODO: Create a preprocessing version using discrete formulas

        deviation_distance_upper_part = (
            (y_axis_arr[:-2] - y_axis_arr[2:]) * x_axis_arr[1:-1]
            + (x_axis_arr[2:] - x_axis_arr[:-2]) * y_axis_arr[1:-1]
            + (x_axis_arr[:-2] * y_axis_arr[2:] - x_axis_arr[2:] * y_axis_arr[:-2])
        )

        deviation_distance_lower_part = np.sqrt(
            (x_axis_arr[2:] - x_axis_arr[:-2]) ** 2 + (y_axis_arr[2:] - y_axis_arr[:-2]) ** 2
        )

        deviation_distance = np.zeros_like(deviation_distance_lower_part)

        np.divide(
            deviation_distance_upper_part,
            deviation_distance_lower_part,
            out=deviation_distance,
            where=deviation_distance_lower_part != 0
        )

        deviation_distance = np.concatenate(([0], deviation_distance, [0]))
        self._extracted_features["deviation_distance"] = deviation_distance

        # 6. Straightness, Efficiency
        straightness = np.zeros_like(self._diff_time_arr)
        np.divide(
            self._traveled_distance,
            self.curve_length,
            out=straightness,
            where=self.curve_length != 0
        )
        self._extracted_features["straightness"] = straightness


        # 7. Jitter
        smoothed_x_axis = np.convolve(x_axis_arr, np.ones(window_size)/window_size, mode='same')
        smoothed_y_axis = np.convolve(y_axis_arr, np.ones(window_size)/window_size, mode='same')

        diff_smoothed_x_axis = np.concatenate(([0], np.diff(smoothed_x_axis)))
        diff_smoothed_y_axis = np.concatenate(([0], np.diff(smoothed_y_axis)))

        smoothed_path_length = np.sqrt(
            diff_smoothed_x_axis ** 2 + diff_smoothed_y_axis ** 2
        )

        jitter = np.zeros_like(self._diff_time_arr)
        np.divide(
            self._traveled_distance,
            smoothed_path_length,
            out=jitter,
            where=smoothed_path_length != 0
        )

        self._extracted_features["jitter"] = jitter

    def _calculate_speed_features(self):
        # 8. Velocity, Speed
        self._speed = np.zeros_like(self._diff_time_arr)

        np.divide(
            self._traveled_distance,
            self._diff_time_arr,
            out=self._speed,
            where=self._diff_time_arr != 0
        )

        self._extracted_features["speed"] = self._speed

        # 9. Horizontal Speed
        self._horizontal_speed = np.zeros_like(self._diff_time_arr)

        np.divide(
            self._diff_x_axis_arr,
            self._diff_time_arr,
            out=self._horizontal_speed,
            where=self._diff_time_arr != 0
        )

        self._extracted_features["horizontal_speed"] = self._horizontal_speed

        # 10. Vertical Speed
        self._vertical_speed = np.zeros_like(self._diff_time_arr)

        np.divide(
            self._diff_y_axis_arr,
            self._diff_time_arr,
            out=self._vertical_speed,
            where=self._diff_time_arr != 0
        )

        self._extracted_features["vertical_speed"] = self._vertical_speed

    def _calculate_acceleration_features(self):
        # 11. Horizontal Acceleration
        diff_x_speed_arr = np.concatenate(([0], np.diff(self._horizontal_speed)))

        self._horizontal_acceleration = np.zeros_like(self._diff_time_arr)

        np.divide(
            diff_x_speed_arr,
            self._diff_time_arr,
            out=self._horizontal_acceleration,
            where=self._diff_time_arr != 0
        )

        self._extracted_features["horizontal_acceleration"] = self._horizontal_acceleration

        # 12. Vertical Acceleration
        diff_y_speed_arr = np.concatenate(([0], np.diff(self._vertical_speed)))

        self._vertical_acceleration = np.zeros_like(self._diff_time_arr)

        np.divide(
            diff_y_speed_arr,
            self._diff_time_arr,
            out=self._vertical_acceleration,
            where=self._diff_time_arr != 0
        )

        self._extracted_features["vertical_acceleration"] = self._vertical_acceleration

        # 13. Acceleration
        diff_speed_arr = np.concatenate(([0], np.diff(self._speed)))

        self._acceleration = np.zeros_like(self._diff_time_arr)

        np.divide(
            diff_speed_arr,
            self._diff_time_arr,
            out=self._acceleration,
            where=self._diff_time_arr != 0
        )

        self._extracted_features["acceleration"] = self._acceleration

        # 14. Average Speed against distance
        avg_speed_against_distance = np.zeros_like(self.curve_length)
        cumulative_speed_avg = np.cumsum(self._speed) / np.arange(1, len(self._speed) + 1)

        np.divide(
            cumulative_speed_avg,
            self.curve_length,
            out=avg_speed_against_distance,
            where=self.curve_length != 0
        )

        self._extracted_features["avg_speed_against_distance"] = avg_speed_against_distance

        # 15. Horizontal Acceleration against Resultant Acceleration
        horizontal_acceleration_against_resultant_acceleration = np.zeros_like(diff_speed_arr)

        np.divide(
            diff_x_speed_arr,
            diff_speed_arr,
            out=horizontal_acceleration_against_resultant_acceleration,
            where=diff_speed_arr != 0
        )

        self._extracted_features["x_acceleration_vs_resultant"] = horizontal_acceleration_against_resultant_acceleration

        # 16. Vertical Acceleration against Resultant Acceleration
        vertical_acceleration_against_resultant_acceleration = np.zeros_like(diff_speed_arr)

        np.divide(
            diff_y_speed_arr,
            diff_speed_arr,
            out=vertical_acceleration_against_resultant_acceleration,
            where=diff_speed_arr != 0
        )

        self._extracted_features["y_acceleration_vs_resultant"] = vertical_acceleration_against_resultant_acceleration

        # 17. Average X Acceleration against distance
        avg_x_acc_against_distance = np.zeros_like(self.curve_length)
        cumulative_x_acc_avg = np.cumsum(self._horizontal_acceleration) / np.arange(1, len(self._horizontal_acceleration) + 1)

        np.divide(
            cumulative_x_acc_avg,
            self.curve_length,
            out=avg_x_acc_against_distance,
            where=self.curve_length != 0
        )

        self._extracted_features["avg_x_acc_against_distance"] = avg_x_acc_against_distance

        # 18. Average Y Acceleration against distance
        avg_y_acc_against_distance = np.zeros_like(self.curve_length)
        cumulative_y_acc_avg = np.cumsum(self._vertical_acceleration) / np.arange(1, len(self._vertical_acceleration) + 1)

        np.divide(
            cumulative_y_acc_avg,
            self.curve_length,
            out=avg_y_acc_against_distance,
            where=self.curve_length != 0
        )

        self._extracted_features["avg_y_acc_against_distance"] = avg_y_acc_against_distance

    def _calculate_angle_features(self):
        # 19. Tangential Speed
        tangential_speed = np.sqrt(
            np.power(self._horizontal_speed, 2)
            + np.power(self._vertical_speed, 2)
        )

        self._extracted_features["tangential_speed"] = tangential_speed

        # 20. Tangential Acceleration
        tangential_acceleration = np.sqrt(
            np.power(self._horizontal_acceleration, 2)
            + np.power(self._vertical_acceleration, 2)
        )

        self._extracted_features["tangential_acceleration"] = tangential_acceleration

        # 21. Tangential Jerk
        diff_tang_acc_arr = np.concatenate(([0], np.diff(tangential_acceleration)))
        tangential_jerk = np.zeros_like(self._diff_time_arr)

        np.divide(
            diff_tang_acc_arr,
            self._diff_time_arr,
            out=tangential_jerk,
            where=self._diff_time_arr != 0
        )

        self._extracted_features["tangential_jerk"] = tangential_jerk

        # 22. Angle of movement
        angle = np.zeros_like(self._diff_time_arr)
        diff_tangential_acc = np.concatenate(([0], np.diff(tangential_acceleration)))

        np.divide(
            diff_tangential_acc,
            self._diff_time_arr,
            out=angle,
            where=self._diff_time_arr != 0
        )

        self._extracted_features["angle"] = angle

        # 23. Rate of curvature
        # TODO

        # 24. Total Angles
        total_angles = np.cumsum(angle)
        self._extracted_features["total_angles"] = total_angles

        # 25. Regularity
        # TODO

        # 26. Trajectory of Center of Mas
        # TODO

        # 27. Scattering Coefficient
        # TODO

        # 28. Curvature Velocity
        curvature_velocity = np.zeros_like(tangential_acceleration)
        np.divide(
            tangential_jerk,
            np.power(1 + np.power(tangential_acceleration, 2), 3 / 2),
        )

        self._extracted_features["curvature_velocity"] = curvature_velocity

        # 29. Central Moments
        # TODO

        # 30. Self-Intersection
        # TODO

        # 31. Angle Feature (Law of cosines)
        # TODO

        # 32. Acceleration Beginning time
        # Calculated in the split df (end)

        # 33. Skewness (third moment)
        # TODO

        # 34. Kurtosis (fourth moment)
        # TODO
"""
A high-performance, vectorized implementation for extracting behavioral biometric features from mouse trajectory data.
"""
from pathlib import Path

from scipy.interpolate import interp1d
from typing import Dict
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BasePreprocessor(ABC):
    """
    Extract behavioral biometric features from mouse trajectory data.

    Features include:
    - Spatial: distances, deviations, curvature
    - Temporal: elapsed time, pauses
    - Kinematic: velocity, acceleration, jerk
    - Statistical: moments, asymmetry, kurtosis

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


class MinecraftPreprocessor(BasePreprocessor):
    """
    Extract behavioral biometric features from mouse trajectory following the guide
    in the article: *Continuous Authentication Using Mouse Movements, Machine Learning, and Minecraft*
    """

    # Number of trajectory points to group together for feature extraction.
    # The trajectory will be divided into non-overlapping windows of this size.
    AMOUNT_OF_LINES_IN_SEQUENCE = 10

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

        dataframes_with_features_by_users = {}

        for user_id, dataframe in dataframes_by_users.items():
            general_features_df = self._extract_general_features_from_df(dataframe)
            statistical_df = self._extract_statistical_info_from_df(general_features_df)
            dataframes_with_features_by_users[user_id] = statistical_df

            logger.info(f"User {user_id} statistical features extracted")
            if is_debug:
                file_path_str = f"../../datasets/features/minecraft/user{user_id}.parquet"

                file = Path(file_path_str)
                file.unlink(missing_ok=True)

                statistical_df.to_parquet(file_path_str, index=False)

        return dataframes_with_features_by_users

    def _extract_statistical_info_from_df(self, general_features_df: pd.DataFrame) -> pd.DataFrame:
        grouped_by_df = general_features_df.groupby(general_features_df.index // self.AMOUNT_OF_LINES_IN_SEQUENCE)

        statistics_extracted_arr = []
        last_value_previous_group = None

        for _, i_df in grouped_by_df:
            is_current_line_unique = (
                i_df[["x", "y"]]
                .ne(i_df[["x", "y"]].shift())
                .any(axis=1)
            )

            if last_value_previous_group is not None and i_df[["x","y"]].iloc[0].equals(last_value_previous_group):
                is_current_line_unique.iloc[0] = False

            i_df_clean = i_df[is_current_line_unique]

            if not i_df_clean.empty:
                last_value_previous_group = i_df_clean[["x", "y"]].iloc[-1]

            # Maybe will be useful later
            # if remove_duplicate:
            #     is_current_line_unique = (i_df[["x", "y"]] != i_df[["x", "y"]].shift()).any(axis=1)
            #
            #     if last_value_previous_group is not None and i_df[["x","y"]].iloc[0].equals(last_value_previous_group):
            #         is_current_line_unique.iloc[0] = False
            #
            #     i_df_clean = i_df[is_current_line_unique]
            #
            #     if not i_df_clean.empty:
            #         last_value_previous_group = i_df_clean[["x", "y"]].iloc[-1]
            # else:
            #     i_df_clean = i_df

            i_df_statistics = {}

            columns = [col for col in i_df_clean.columns.tolist() if col not in  {"x", "y", "timestamp"}]

            for col_name in columns:
                i_df_statistics[f"mean_{col_name}"] = i_df_clean[col_name].mean()
                i_df_statistics[f"std_{col_name}"] = i_df_clean[col_name].std()
                i_df_statistics[f"max_{col_name}"] = i_df_clean[col_name].max()
                i_df_statistics[f"min_{col_name}"] = i_df_clean[col_name].min()

            #TODO: check if I should do this in the clean df or the original (with duplicates)
            if len(i_df_clean) >= 2:
                i_df_statistics["acc_beginning_time"] = i_df_clean["timestamp"].iat[1] - i_df_clean["timestamp"].iat[0]
            else:
                i_df_statistics["acc_beginning_time"] = 0

            statistics_extracted_arr.append(i_df_statistics)

        return pd.DataFrame(statistics_extracted_arr)

    def _extract_general_features_from_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all movement features from a trajectory.

        :return: Dataframe containing all extracted features with descriptive names
        """

        # Convert to numpy arrays for performance
        x = dataframe["x"].values.astype(float)
        y = dataframe["y"].values.astype(float)
        t = dataframe["timestamp"].values.astype(float)

        general_features = {
            "x": dataframe["x"],
            "y": dataframe["y"],
            "timestamp": dataframe["timestamp"],
        }

        general_features.update(self._extract_kinematic_features(x, y, t))
        general_features.update(self._extract_temporal_features(t))
        general_features.update(self._extract_curvature_features(x, y, t))

        #TODO: Adjust the following methods

        #TODO: Add the others features from the sheets that are not present in the Minecraft official extraction

        # general_features.update(self._extract_spatial_features(x, y))
        # general_features.update(self._extract_statistical_features(x, y))

        return pd.DataFrame(general_features)

    def _extract_spatial_features(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Extract spatial geometry features.

        These features describe the geometric properties of the mouse trajectory,
        including path length, straightness, and deviations.
        """
        features = {}

        # Calculate segment lengths
        dx = np.diff(x)
        dy = np.diff(y)
        segments = np.sqrt(dx**2 + dy**2)

        # Distância Percorrida (Di) - Total path length
        features['distancia_percorrida'] = np.sum(segments)

        # Comprimento da Curva / Distância Real - Total path length (same as above)
        features['comprimento_curva'] = features['distancia_percorrida']

        # Distância Reta (Di) - Euclidean distance from start to end
        dist_reta = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        features['distancia_reta'] = dist_reta

        # Deslocamento do Movimento - Difference between curved and straight path
        features['deslocamento_movimento'] = features['distancia_percorrida'] - dist_reta

        # Distância de Desvio - Maximum perpendicular distance from straight line
        features['distancia_desvio'] = self._calculate_max_deviation(x, y)

        # Retidão / Eficiência - Ratio of straight distance to path length
        # Measures how straight the trajectory is (1.0 = perfectly straight)
        if features['distancia_percorrida'] > 0:
            features['retidao_eficiencia'] = dist_reta / features['distancia_percorrida']
        else:
            features['retidao_eficiencia'] = 0.0

        # Comprimento Interpolado (ln) - Smoothed path length
        features['comprimento_interpolado'] = self._calculate_interpolated_length(x, y)

        # Tremor (Jitter) - Ratio of interpolated to real path
        # Lower values indicate smoother movements
        if features['distancia_percorrida'] > 0:
            features['tremor_jitter'] = features['comprimento_interpolado'] / features['distancia_percorrida']
        else:
            features['tremor_jitter'] = 1.0

        # Regularidade - Measure of path regularity
        # Combines mean and std of segment lengths relative to straight distance
        if dist_reta > 0:
            mu_d = np.mean(segments)
            sigma_d = np.std(segments)
            features['regularidade'] = (mu_d + sigma_d) / dist_reta
        else:
            features['regularidade'] = 0.0

        # Auto-Interseção - Number of self-intersections
        features['auto_intersecao'] = self._count_self_intersections(x, y)

        return features

    def _extract_temporal_features(self, t: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract temporal features.

        These features describe timing characteristics of the mouse movement.
        """

        # Elapsed time / Mouse digraph
        temporal_features = {
            'elapsed_time': np.concatenate(([0], np.diff(t)))
        }


        return temporal_features

    def _extract_kinematic_features(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Dict[str, float]:
        """
        Extract kinematic features (velocity, acceleration, jerk).

        These features describe the motion characteristics of the mouse cursor.

        :param x: the X coordinate array
        :param y: the Y coordinate array
        :param t: the timestamp array
        """
        #TODO: check literature to see if I should use the absolute values of x - x-1 and y - y-1 (std is too big)
        motion_features = {}

        # Calculate time differences
        time_differences_arr = np.concatenate(([0], np.diff(t)))

        # Calculate spatial differences
        x_differences_arr = np.concatenate(([0], np.diff(x)))
        y_differences_arr = np.concatenate(([0], np.diff(y)))

        # The first value of the dataframe is always zero
        #TODO: Probably there is other feature_extractions that doesnt make the first zero or make the last also zero. Check if is a good idea to parameterize this options

        # Speed magnitude {sqrt(x_differences_arr*2 + y_differences_arr*2) / time_differences_arr}
        speed_arr = np.zeros_like(time_differences_arr)
        np.divide(np.sqrt(x_differences_arr**2 + y_differences_arr**2), time_differences_arr, out = speed_arr, where = time_differences_arr != 0)
        motion_features['speed'] = speed_arr

        # Horizontal speed (x-axis) {x_differences_arr / time_differences_arr}
        x_speed_arr = np.zeros_like(time_differences_arr)
        #TODO: check my own audio recording in my personal whatsapp group. Talk to professor about it (duplicates interfering in timestamp difference)
        np.divide(x_differences_arr, time_differences_arr, out = x_speed_arr, where = time_differences_arr != 0)
        motion_features['speed_x'] = x_speed_arr

        # Vertical speed (y-axis) {y_differences_arr / time_differences_arr}
        y_speed_arr = np.zeros_like(time_differences_arr)
        np.divide(y_differences_arr, time_differences_arr, out = y_speed_arr, where = time_differences_arr != 0)
        motion_features['speed_y'] = y_speed_arr

        #TODO: finish the motion feature extraction

        # Acceleration magnitude {speed_difference_arr / time_differences_arr}
        #TODO: talk to professor: in the original dataset, they calculate the acceleration wrong: dt / dv
        speed_difference_arr = np.concatenate(([0], np.diff(speed_arr)))

        acceleration_arr = np.zeros_like(time_differences_arr)
        np.divide(speed_difference_arr, time_differences_arr, out = acceleration_arr, where = time_differences_arr != 0)
        motion_features['acceleration'] = acceleration_arr

        # Horizontal Acceleration  (x-axis) {x_speed_difference_arr / time_differences_arr}
        x_speed_difference_arr = np.concatenate(([0], np.diff(x_speed_arr)))

        x_acceleration = np.zeros_like(time_differences_arr)
        np.divide(x_speed_difference_arr, time_differences_arr, out = x_acceleration, where = time_differences_arr != 0)
        motion_features['acceleration_x'] = x_acceleration

        # Vertical Acceleration  (y-axis) {y_speed_difference_arr / time_differences_arr}
        y_speed_difference_arr = np.concatenate(([0], np.diff(y_speed_arr)))

        y_acceleration = np.zeros_like(time_differences_arr)
        np.divide(y_speed_difference_arr, time_differences_arr, out = y_acceleration, where = time_differences_arr != 0)
        motion_features['acceleration_y'] = y_acceleration

        y_acceleration_difference_arr = np.concatenate(([0], np.diff(y_acceleration)))
        x_acceleration_difference_arr = np.concatenate(([0], np.diff(x_acceleration)))

        acceleration_difference_arr = np.concatenate(([0], np.diff(acceleration_arr)))

        jerk = np.zeros_like(time_differences_arr)
        np.divide(acceleration_difference_arr, time_differences_arr, out = jerk, where = time_differences_arr != 0)
        motion_features['jerk'] = jerk

        curvatures = np.zeros_like(time_differences_arr)
        curvature_upper_part = x_speed_difference_arr*y_acceleration_difference_arr - y_speed_difference_arr*x_acceleration_difference_arr
        curvature_lower_part = np.power(x_speed_difference_arr**2 + y_speed_difference_arr**2, 3/2)

        np.divide(curvature_upper_part, curvature_lower_part, out = curvatures, where = curvature_lower_part != 0)
        motion_features['curvature'] = curvatures

        return motion_features

    def _extract_curvature_features(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Dict[str, float]:
        """
        Extract curvature-related features.

        These features describe how the trajectory curves and changes direction.
        """
        curvature_features = {}
        # Calculate time differences

        # Calculate spatial differences
        x_differences_arr = np.concatenate(([0], np.diff(x)))
        y_differences_arr = np.concatenate(([0], np.diff(y)))

        # Movement angle - Tangent angle relative to x-axis
        slopes = np.zeros_like(x_differences_arr)
        np.divide(y_differences_arr, x_differences_arr, out = slopes, where = x_differences_arr != 0)

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

    def _extract_statistical_features(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical moments and distribution features.

        These features describe the statistical distribution of trajectory points.
        """
        features = {}

        # Center the coordinates around their mean
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        # Calculate combined position variance (distance from center of mass)
        position_variance = x_centered**2 + y_centered**2

        # === CENTRAL MOMENTS ===
        # Momentos Centrais (μn) - Central moments about the mean

        features['momento_central_2'] = np.mean(position_variance)
        features['momento_central_3'] = np.mean(position_variance**1.5)
        features['momento_central_4'] = np.mean(position_variance**2)

        # === ASYMMETRY (SKEWNESS) ===
        # Assimetria (Skewness) - 3rd standardized moment
        # Measures asymmetry of the distribution

        # if len(x) > 2:
        #     features['assimetria_x'] = stats.skew(x)
        #     features['assimetria_y'] = stats.skew(y)
        #     features['assimetria_combinada'] = (features['assimetria_x'] + features['assimetria_y']) / 2

        # === KURTOSIS ===
        # Curtose (Kurtosis) - 4th standardized moment
        # Measures "tailedness" of the distribution

        # if len(x) > 3:
        #     features['curtose_x'] = stats.kurtosis(x)
        #     features['curtose_y'] = stats.kurtosis(y)
        #     features['curtose_combinada'] = (features['curtose_x'] + features['curtose_y']) / 2

        return features

    def _calculate_max_deviation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate maximum perpendicular distance from the straight line.

        This is essentially the maximum deviation from the straight path
        between start and end points.
        """
        if len(x) < 3:
            return 0.0

        # Line from start to end
        x0, y0 = x[0], y[0]
        x1, y1 = x[-1], y[-1]

        # Vector of the line
        dx_line = x1 - x0
        dy_line = y1 - y0
        line_length = np.sqrt(dx_line**2 + dy_line**2)

        if line_length == 0:
            return 0.0

        # Calculate perpendicular distances for all points
        # Using the formula: d = |ax + by + c| / sqrt(a² + b²)
        distances = np.abs(dy_line * (x - x0) - dx_line * (y - y0)) / line_length

        return np.max(distances)

    def _calculate_interpolated_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate smoothed/interpolated path length.

        Uses spline interpolation to create a smoothed version of the trajectory
        and calculates its length. This helps reduce the effect of noise.
        """
        if len(x) < 3:
            return 0.0

        # Use quadratic spline interpolation for smoothing
        t_original = np.arange(len(x))
        t_smooth = np.linspace(0, len(x) - 1, len(x) * 2)

        fx = interp1d(t_original, x, kind='quadratic', fill_value='extrapolate')
        fy = interp1d(t_original, y, kind='quadratic', fill_value='extrapolate')

        x_smooth = fx(t_smooth)
        y_smooth = fy(t_smooth)

        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)

        return np.sum(np.sqrt(dx**2 + dy**2))

    def _count_self_intersections(self, x: np.ndarray, y: np.ndarray) -> int:
        """
        Count the number of self-intersections in the trajectory.

        A self-intersection occurs when the trajectory crosses itself.
        This uses a simple O(n²) algorithm, so it's skipped for large trajectories.
        """
        n = len(x)

        # Skip for large trajectories due to computational cost
        if n > 100:
            return 0

        count = 0

        # Check each pair of non-adjacent segments
        for i in range(n - 1):
            for j in range(i + 2, n - 1):
                if self._segments_intersect(x[i], y[i], x[i+1], y[i+1],
                                           x[j], y[j], x[j+1], y[j+1]):
                    count += 1

        return count

    def _segments_intersect(self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
                           x3: np.ndarray, y3: np.ndarray, x4: np.ndarray, y4: np.ndarray) -> bool:
        """
        Check if two line segments intersect.

        Uses the counterclockwise (CCW) method to determine intersection.
        """
        def ccw(ax, ay, bx, by, cx, cy):
            """Check if three points are in counterclockwise order."""
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

        # Two segments intersect if the endpoints of each segment are on
        # opposite sides of the other segment
        return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
                ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))

    def _calculate_cosine_angles(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate angles between consecutive segments using law of cosines.

        For each triplet of consecutive points, calculates the angle formed.
        """
        if len(x) < 3:
            return np.array([])

        # Vectors between consecutive points
        v1x = x[1:-1] - x[:-2]
        v1y = y[1:-1] - y[:-2]
        v2x = x[2:] - x[1:-1]
        v2y = y[2:] - y[1:-1]

        # Lengths of vectors
        len1 = np.sqrt(v1x**2 + v1y**2)
        len2 = np.sqrt(v2x**2 + v2y**2)

        # Dot product
        dot_product = v1x * v2x + v1y * v2y

        # Avoid division by zero
        denominator = len1 * len2
        denominator = np.where(denominator == 0, 1e-10, denominator)

        # Cosine of angle (clipped to [-1, 1] for numerical stability)
        cos_angle = np.clip(dot_product / denominator, -1.0, 1.0)

        # Convert to angle in radians
        angles = np.arccos(cos_angle)

        return angles
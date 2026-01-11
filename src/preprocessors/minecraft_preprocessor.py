"""
Preprocessing the features following the `Continuous Authentication Using Mouse Movements,
Machine Learning, and Minecraft` article.
"""
from src.preprocessors import BasePreprocessor
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
                   curvature_threshold: float = 0.0005) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all the dataframes by user

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
            if self.is_debug:
                file_path_str = f"../datasets/features/user{user_id}.parquet"

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

            if (last_value_previous_group is not None
                    and i_df[["x","y"]].iloc[0].equals(last_value_previous_group)):
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
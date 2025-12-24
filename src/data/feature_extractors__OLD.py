"""
A high-performance, vectorized implementation for extracting behavioral biometric
features from mouse trajectory data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)

class MouseDynamicsExtractor:
    """
    Extract behavioral biometric features from mouse trajectory data.

    Features include:
    - Spatial: distances, deviations, amplitude
    - Temporal: elapsed time, maximum movement time, time interval
    - Kinematic: velocity, acceleration, jerk
    - Form: movement efficiency, curvature, angle

    The extractor can process trajectories in sequential windows, extracting features
    from each window to create a sequence of feature vectors.
    """

    def __init__(self,
                 amount_of_lines_in_sequence: int = 10,
                 curvature_threshold: float = 0.0005):
        """
        Initialize the feature extractor.

        :param amount_of_lines_in_sequence: Number of trajectory points to group together
                                        for feature extraction. The trajectory will be
                                        divided into non-overlapping windows of this size.
                                        Default: 10 points per window.

        :param curvature_threshold: Threshold for detecting critical curvature points (THc)
                               Used in some advanced curvature analysis features.
        """
        self.amount_of_lines_in_sequence = amount_of_lines_in_sequence
        self.curvature_threshold = curvature_threshold

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all mouse dynamics features from a trajectory.

        :param df: DataFrame containing mouse trajectory data
        :return Dictionary containing all extracted features with descriptive names
        :raise ValueError: If trajectory has fewer than 3 points
        """
        #TODO: receive the preprocessing as parameter and adjust the amount_of_lines_in_sequence

        # Calculate number of windows
        n_points = len(df)
        window_size = self.amount_of_lines_in_sequence
        n_windows = n_points // window_size

        if n_windows == 0:
            # If trajectory is shorter than window size, use entire trajectory
            features = self._extract_features_from_window(df)
            return pd.DataFrame([features])

        # Extract features from each window
        all_features = []

        for window_id in range(n_windows):
            start_idx = window_id * window_size
            end_idx = start_idx + window_size

            # Extract window
            window_df = df.iloc[start_idx:end_idx].copy()

            # Skip windows with insufficient points
            if len(window_df) < 3:
                continue

            try:
                # Extract features from this window
                features = self._extract_features_from_window(window_df)
                all_features.append(features)

            except Exception as e:
                logger.warning(f"Failed to extract features from window {window_id}: {e}")
                continue

        # Handle remaining points (if any)
        remaining_start = n_windows * window_size
        if remaining_start < n_points:
            remaining_df = df.iloc[remaining_start:].copy()

            # Only process if we have at least 3 points
            if len(remaining_df) >= 3:
                try:
                    features = self._extract_features_from_window(remaining_df)
                    all_features.append(features)
                except Exception as e:
                    logging.warning(f"Failed to extract features from remaining window: {e}")

        # Convert to DataFrame
        if not all_features:
            raise ValueError("No features could be extracted from any window")

        features_df = pd.DataFrame(all_features)
        return features_df

    def _extract_features_from_window(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from a single window of trajectory data.

        Args:
            df: DataFrame containing a window of trajectory data

        Returns:
            Dictionary containing all extracted features
        """
        # Convert to numpy arrays for performance
        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)
        t = df["timestamp"].values.astype(float)

        # Normalize time to start at 0
        t = t - t[0]

        # Initialize features dictionary
        features = {}

        # === SPATIAL FEATURES ===
        features.update(self._extract_spatial_features(x, y))

        # === TEMPORAL FEATURES ===
        features.update(self._extract_temporal_features(t))

        # === KINEMATIC FEATURES ===
        features.update(self._extract_kinematic_features(x, y, t))

        # === STATISTICAL FEATURES ===
        features.update(self._extract_statistical_features(x, y))

        # === CURVATURE FEATURES ===
        features.update(self._extract_curvature_features(x, y, t))

        return features

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

    def _extract_temporal_features(self, t: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features.

        These features describe timing characteristics of the mouse movement.
        """
        features = {}

        # Tempo Decorrido / Dígrafo do Mouse - Total elapsed time
        features['tempo_decorrido'] = t[-1] - t[0] if len(t) > 1 else 0.0

        # Tempo Inicial de Aceleração - Time until acceleration starts
        # (approximated as time to second point)
        features['tempo_inicial_aceleracao'] = t[1] - t[0] if len(t) > 1 else 0.0

        return features

    def _extract_kinematic_features(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Dict[str, float]:
        """
        Extract kinematic features (velocity, acceleration, jerk).

        These features describe the motion characteristics of the mouse cursor.
        """
        features = {}

        # Calculate time differences (avoid division by zero)
        dt = np.diff(t)
        dt = np.where(dt == 0, 1e-10, dt)

        # Calculate spatial differences
        dx = np.diff(x)
        dy = np.diff(y)

        # === VELOCITY FEATURES ===

        # Velocidade (V) - Speed magnitude
        velocities = np.sqrt(dx**2 + dy**2) / dt
        features['velocidade_media'] = np.mean(velocities)
        features['velocidade_max'] = np.max(velocities)
        features['velocidade_std'] = np.std(velocities)

        # Velocidade Horizontal (x'(t)) - Velocity in x direction
        vx = dx / dt
        features['velocidade_horizontal_media'] = np.mean(vx)
        features['velocidade_horizontal_std'] = np.std(vx)

        # Velocidade Vertical (y'(t)) - Velocity in y direction
        vy = dy / dt
        features['velocidade_vertical_media'] = np.mean(vy)
        features['velocidade_vertical_std'] = np.std(vy)

        # === ACCELERATION FEATURES ===

        if len(velocities) > 1:
            # Aceleração (A) - Rate of change of speed
            accelerations = np.diff(velocities) / dt[1:]
            features['aceleracao_media'] = np.mean(accelerations)
            features['aceleracao_max'] = np.max(np.abs(accelerations))
            features['aceleracao_std'] = np.std(accelerations)

            # Aceleração Horizontal (x''(t)) - Acceleration in x direction
            ax = np.diff(vx) / dt[1:]
            features['aceleracao_horizontal_media'] = np.mean(ax)
            features['aceleracao_horizontal_std'] = np.std(ax)

            # Aceleração Vertical (y''(t)) - Acceleration in y direction
            ay = np.diff(vy) / dt[1:]
            features['aceleracao_vertical_media'] = np.mean(ay)
            features['aceleracao_vertical_std'] = np.std(ay)

            # === JERK FEATURES (rate of change of acceleration) ===

            if len(accelerations) > 1:
                jerk = np.diff(accelerations) / dt[2:]
                features['saculejo_tangencial_media'] = np.mean(jerk)
                features['saculejo_tangencial_std'] = np.std(jerk)

        # === DISTANCE-BASED VELOCITY AND ACCELERATION ===

        distances = np.sqrt(dx**2 + dy**2)
        cumulative_dist = np.concatenate([[0], np.cumsum(distances)])

        if len(cumulative_dist) > 2:
            # Velocidade em Função da Distância
            features['velocidade_funcao_distancia_media'] = np.mean(velocities)

            # Aceleração X em Função da Distância
            if len(vx) > 1:
                dist_segments = distances[:-1]
                dist_segments = np.where(dist_segments == 0, 1e-10, dist_segments)
                ax_dist = np.diff(vx) / dist_segments
                features['aceleracao_x_funcao_distancia_media'] = np.mean(ax_dist)

            # Aceleração Y em Função da Distância
            if len(vy) > 1:
                dist_segments = distances[:-1]
                dist_segments = np.where(dist_segments == 0, 1e-10, dist_segments)
                ay_dist = np.diff(vy) / dist_segments
                features['aceleracao_y_funcao_distancia_media'] = np.mean(ay_dist)

        # === TANGENTIAL MOTION ===

        # Velocidade Tangencial (Vt) - Speed along the curve
        tangential_velocity = velocities
        features['velocidade_tangencial_media'] = np.mean(tangential_velocity)
        features['velocidade_tangencial_std'] = np.std(tangential_velocity)

        # Aceleração Tangencial (A) - Rate of change of tangential velocity
        if len(tangential_velocity) > 1:
            tangential_acceleration = np.diff(tangential_velocity) / dt[1:]
            features['aceleracao_tangencial_media'] = np.mean(tangential_acceleration)
            features['aceleracao_tangencial_std'] = np.std(tangential_acceleration)

        return features

    def _extract_form_features(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Dict[str, float]:
        """
        Extract curvature-related features.

        These features describe how the trajectory curves and changes direction.
        Curvature is calculated using the formula: κ = |x'y'' - y'x''| / (x^2 + y^2)^(3/2)
        """
        features = {}

        # Calculate time differences (avoid division by zero)
        dt = np.diff(t)
        dt = np.where(dt == 0, 1e-10, dt)

        if len(x) < 3:
            # Not enough points for curvature calculation
            features['curvatura_media'] = 0.0
            features['velocidade_curvatura_media'] = 0.0
            features['taxa_curvatura_media'] = 0.0
            features['angulo_movimento_medio'] = 0.0
            features['raio_curvatura_medio'] = 0.0
            features['velocidade_angular_media'] = 0.0
            features['angulos_totais_energia_curvatura'] = 0.0
            return features

        # First derivatives (velocity components)
        dx = np.diff(x)
        dy = np.diff(y)

        # Second derivatives (acceleration components)
        if len(x) > 2:
            ddx = np.diff(dx) / dt[1:]
            ddy = np.diff(dy) / dt[1:]

            # Velocity components at points where we have acceleration
            vx = dx[1:] / dt[1:]
            vy = dy[1:] / dt[1:]

            # Curvature calculation: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
            numerator = np.abs(vx * ddy - vy * ddx)
            denominator = (vx**2 + vy**2)**1.5
            denominator = np.where(denominator == 0, 1e-10, denominator)

            curvature = numerator / denominator

            # === CURVATURE STATISTICS ===

            features['curvatura_media'] = np.mean(curvature)
            features['curvatura_std'] = np.std(curvature)
            features['curvatura_max'] = np.max(curvature)

            # Raio de Curvatura médio (radius of curvature = 1/κ)
            radii = 1.0 / (curvature + 1e-10)
            # Filter out extreme values for stability
            features['raio_curvatura_medio'] = np.mean(radii[radii < 1e8])

            # Velocidade de Curvatura (Vcurve) - Rate of change of curvature magnitude
            if len(curvature) > 1:
                vcurve = np.sqrt(curvature[:-1]**2 + curvature[1:]**2)
                features['velocidade_curvatura_media'] = np.mean(vcurve)

            # Taxa de Curvatura - Time derivative of curvature
            if len(curvature) > 1:
                dcurvature = np.diff(curvature) / dt[2:]
                features['taxa_curvatura_media'] = np.mean(np.abs(dcurvature))

            # Ângulos Totais / Energia de Curvatura - Sum of absolute curvatures
            features['angulos_totais_energia_curvatura'] = np.sum(np.abs(curvature))

        # === ANGULAR FEATURES ===

        # Ângulo do Movimento (α) - Tangent angle relative to x-axis
        angles = np.arctan2(dy, dx)
        features['angulo_movimento_medio'] = np.mean(angles)
        features['angulo_movimento_std'] = np.std(angles)

        # Velocidade Angular (ωi) - Rate of change of angle
        if len(angles) > 1:
            angular_velocity = np.diff(angles) / dt[1:]
            # Handle angle wrapping (convert to [-π, π] range)
            angular_velocity = np.arctan2(np.sin(angular_velocity), np.cos(angular_velocity))
            features['velocidade_angular_media'] = np.mean(np.abs(angular_velocity))
            features['velocidade_angular_std'] = np.std(angular_velocity)

        # Ângulo (γ) - Lei dos Cossenos - Angle between consecutive segments
        if len(x) > 2:
            angles_cosine = self._calculate_cosine_angles(x, y)
            features['angulo_lei_cossenos_medio'] = np.mean(angles_cosine)
            features['angulo_lei_cossenos_std'] = np.std(angles_cosine)

        return features


    #TODO: Fix
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

        if len(x) > 2:
            features['assimetria_x'] = stats.skew(x)
            features['assimetria_y'] = stats.skew(y)
            features['assimetria_combinada'] = (features['assimetria_x'] + features['assimetria_y']) / 2

        # === KURTOSIS ===
        # Curtose (Kurtosis) - 4th standardized moment
        # Measures "tailedness" of the distribution

        if len(x) > 3:
            features['curtose_x'] = stats.kurtosis(x)
            features['curtose_y'] = stats.kurtosis(y)
            features['curtose_combinada'] = (features['curtose_x'] + features['curtose_y']) / 2

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

        try:
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
        except:
            # Fallback to original length if interpolation fails
            dx = np.diff(x)
            dy = np.diff(y)
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

    def _segments_intersect(self, x1: float, y1: float, x2: float, y2: float,
                           x3: float, y3: float, x4: float, y4: float) -> bool:
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


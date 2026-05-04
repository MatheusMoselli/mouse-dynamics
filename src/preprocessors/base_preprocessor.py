"""
Generic preprocessor for feature extraction and statistical analysis.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging

from src.dto import ExtractionData
from src.utils.log_file import log_dataframe_sessions

logger = logging.getLogger(__name__)

# Columns that carry metadata.
_META_COLS = frozenset({"x", "y", "timestamp", "authentic", "session"})

# The four statistics computed for every feature column.
_STAT_FUNCS = ["mean", "std", "max", "min"]


class BasePreprocessor:
    """
    Base class for all mouse-dynamics feature preprocessors.

    Provides a complete, concrete preprocessing pipeline that subclasses inherit
    without any overriding required.
    """
    _diff_x_axis_arr: np.ndarray = np.array([])
    _diff_y_axis_arr: np.ndarray = np.array([])
    _diff_time_arr: np.ndarray = np.array([])
    _traveled_distance: np.ndarray = np.array([])
    curve_length: np.ndarray = np.array([])

    _speed: np.ndarray = np.array([])
    _horizontal_speed: np.ndarray = np.array([])
    _vertical_speed: np.ndarray = np.array([])

    _acceleration: np.ndarray = np.array([])
    _horizontal_acceleration: np.ndarray = np.array([])
    _vertical_acceleration: np.ndarray = np.array([])

    _extracted_features: dict = {}
    __features_dataframe: pd.DataFrame | None = None

    def __init__(self, is_debug: bool = False, window_size: int = 40):
        """
        :param is_debug: When True, write intermediate DataFrames to parquet.
        :param window_size: Size of the window the preprocessor will use the aggregate features
        """
        self.is_debug = is_debug
        self._window_size = window_size

    def preprocess(self, extraction_data: ExtractionData) -> ExtractionData:
        """
        Preprocess all sessions for every user in extraction_data.

        :param extraction_data: Users with raw session DataFrames
        :return: Same object with session DataFrames replaced by statistics DataFrames
        """
        for user in extraction_data.users:
            logger.info(f"Preprocessing user [TRAINING]: {user.id}")
            user.training_sessions = self._process_sessions(user.training_sessions)

            logger.info(f"Preprocessing user [TEST]: {user.id}")
            user.testing_sessions = self._process_sessions(user.testing_sessions)

            logger.info(f"User {user.id}: statistical features extracted")

            if self.is_debug:
                self._write_debug_files(user)

        return extraction_data

    def _process_sessions(
        self, sessions: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Extract the features and obtain the statistical features for each

        :param sessions: Raw session DataFrames keyed by session name
        :return: New dict with the same keys but statistics DataFrames as values
        """
        result: dict[str, pd.DataFrame] = {}

        for session_name, session_df in sessions.items():
            if session_df.empty:
                logger.warning(f"Session '{session_name}' is empty, skipping.")
                continue
            result[session_name] = self._process_single_session(session_df)

        return result

    def _process_single_session(self, session_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline for a single session DataFrame:
          1. Extract per-point kinematic features
          2. Deduplicate consecutive identical (x, y) points
          3. Slice into non-overlapping windows of _window_size
          4. Compute [mean, std, max, min] per window in one vectorised pass

        :param session_df: Raw DataFrame for a single session
        :return: Statistics DataFrame with one row per window
        """
        authentic = int(session_df["authentic"].iat[0])

        x = session_df["x"].to_numpy(dtype=float)
        y = session_df["y"].to_numpy(dtype=float)
        t = session_df["timestamp"].to_numpy(dtype=float)

        self._initialize_extracted_features_df(session_df["authentic"].to_numpy())
        self._calculate_basic_axis_features(x, y, t)
        self._calculate_speed_features()
        self._calculate_acceleration_features()
        self._calculate_angle_features()

        features_df = self.features_dataframe.copy()

        xy = features_df[["x", "y"]].to_numpy()
        is_unique = np.ones(len(xy), dtype=bool)
        is_unique[1:] = np.any(xy[1:] != xy[:-1], axis=1)
        features_df = features_df.loc[is_unique].reset_index(drop=True)

        if features_df.empty:
            return pd.DataFrame()

        return self._extract_statistics(features_df, authentic)

    def _extract_statistics(
        self, features_df: pd.DataFrame, authentic: int
    ) -> pd.DataFrame:
        """
        Full pipeline of statistics extraction:
          1. Divide features_df into non-overlapping windows of _window_size rows
          2. Compute [mean, std, max, min] for every feature column in a single pass
          3. Compute features that are dependent on _window_size
        :param features_df: Deduplicated per-point feature DataFrame for one session
        :param authentic: Authenticity label (0 or 1) for this session
        :return: Statistics DataFrame with one row per window
        """
        feature_cols = [c for c in features_df.columns if c not in _META_COLS]
        n = len(features_df)
        window_ids = np.arange(n) // self._window_size

        agg_df = features_df[feature_cols].groupby(window_ids).agg(_STAT_FUNCS)

        agg_df.columns = [f"{stat}_{col}" for col, stat in agg_df.columns]
        agg_df = agg_df.reset_index(drop=True)

        traveled = features_df["traveled_distance"].to_numpy()
        speed    = features_df["speed"].to_numpy()
        h_acc    = features_df["horizontal_acceleration"].to_numpy()
        v_acc    = features_df["vertical_acceleration"].to_numpy()
        angle    = features_df["angle"].to_numpy()
        timestamps = features_df["timestamp"].to_numpy()

        window_starts = np.arange(0, n, self._window_size)
        n_windows = len(window_starts)

        curve_length_w        = np.empty(n_windows)
        total_angles_w        = np.empty(n_windows)
        avg_spd_dist_w        = np.zeros(n_windows)
        avg_x_acc_dist_w      = np.zeros(n_windows)
        avg_y_acc_dist_w      = np.zeros(n_windows)
        acc_beginning_time_w  = np.zeros(n_windows)

        for i, start in enumerate(window_starts):
            end = min(start + self._window_size, n)
            sl_traveled = traveled[start:end]
            sl_speed    = speed[start:end]
            sl_h_acc    = h_acc[start:end]
            sl_v_acc    = v_acc[start:end]
            sl_angle    = angle[start:end]
            sl_ts       = timestamps[start:end]

            # 2.  curve_length  — total path length inside the window
            curve_len = sl_traveled.sum()
            curve_length_w[i] = curve_len

            # 24. total_angles  — algebraic sum of direction changes in window
            total_angles_w[i] = sl_angle.sum()

            # 14. avg_speed_against_distance  — mean speed normalised by path length
            if curve_len != 0:
                avg_spd_dist_w[i] = sl_speed.mean() / curve_len

            # 17. avg_x_acc_against_distance
            if curve_len != 0:
                avg_x_acc_dist_w[i] = sl_h_acc.mean() / curve_len

            # 18. avg_y_acc_against_distance
            if curve_len != 0:
                avg_y_acc_dist_w[i] = sl_v_acc.mean() / curve_len

            # 32. acc_beginning_time  — Δt between first two timestamps of window
            if len(sl_ts) > 1:
                acc_beginning_time_w[i] = sl_ts[1] - sl_ts[0]

        agg_df["curve_length"] = curve_length_w[: len(agg_df)]
        agg_df["total_angles"] = total_angles_w[: len(agg_df)]
        agg_df["avg_speed_against_distance"] = avg_spd_dist_w[: len(agg_df)]
        agg_df["avg_x_acc_against_distance"] = avg_x_acc_dist_w[: len(agg_df)]
        agg_df["avg_y_acc_against_distance"] = avg_y_acc_dist_w[: len(agg_df)]
        agg_df["acc_beginning_time"] = acc_beginning_time_w[: len(agg_df)]
        agg_df["authentic"] = authentic

        return agg_df

    @property
    def features_dataframe(self) -> pd.DataFrame:
        """
        Get the internal features dataframe
        :return: __features_dataframe
        """
        if self.__features_dataframe is None:
            self.__features_dataframe = pd.DataFrame(self._extracted_features)
        return self.__features_dataframe

    def _initialize_extracted_features_df(self, authentic_arr: np.ndarray) -> None:
        """
        Reset internal feature state before processing a new session.

        Session identity is tracked externally as the dict key so that features
        are computed on clean per-session arrays.

        :param authentic_arr: Authenticity label array for the session rows
        """
        self.__features_dataframe = None
        self._extracted_features = {"authentic": authentic_arr}

    def _calculate_basic_axis_features(
        self,
        x_axis_arr: np.ndarray,
        y_axis_arr: np.ndarray,
        time_arr: np.ndarray,
    ) -> None:
        self._extracted_features["x"] = x_axis_arr
        self._extracted_features["y"] = y_axis_arr
        self._extracted_features["timestamp"] = time_arr

        self._diff_x_axis_arr = np.concatenate(([0], np.diff(x_axis_arr)))
        self._diff_y_axis_arr = np.concatenate(([0], np.diff(y_axis_arr)))
        self._diff_time_arr = np.concatenate(([0], np.diff(time_arr)))

        window_size = 10

        # 1. Traveled Distance (Di)
        self._traveled_distance = np.sqrt(
            np.power(self._diff_x_axis_arr, 2) + np.power(self._diff_y_axis_arr, 2)
        )
        self._extracted_features["traveled_distance"] = self._traveled_distance

        # 2. Curve length / Real Dist. (Sn)
        # Kept as an internal array for downstream feature calculations.
        # It is NOT stored as a per-point feature because it is a global
        # cumulative sum that would carry cross-window history; the per-window
        # value (sum of traveled_distance within the window) is computed
        # directly in _extract_statistics.
        self.curve_length = np.cumsum(self._traveled_distance)

        # 3. Elapsed Time / Mouse Digraph
        self._extracted_features["elapsed_time"] = self._diff_time_arr

        # 4. Movement Offset
        # Difference between the cumulative path length (curve_length) and the
        # straight-line displacement from the very first point to the current
        # point.  This captures how much the cursor "wandered" relative to the
        # direct route, which is the standard definition used in the literature.
        direct_displacement = np.sqrt(
            (x_axis_arr - x_axis_arr[0]) ** 2 + (y_axis_arr - y_axis_arr[0]) ** 2
        )
        self._extracted_features["movement_offset"] = self.curve_length - direct_displacement

        # 5. Deviation Distance
        # Perpendicular distance from point i to the chord connecting its two
        # neighbours (i-1, i+1).  The cross-product formula gives a signed
        # area; dividing by the chord length yields a signed distance whose
        # sign indicates which side of the chord the point lies on.  Taking
        # the absolute value converts it to a true (non-negative) distance.
        # TODO: Create a preprocessing version using discrete formulas
        deviation_distance_upper_part = (
            (y_axis_arr[:-2] - y_axis_arr[2:]) * x_axis_arr[1:-1]
            + (x_axis_arr[2:] - x_axis_arr[:-2]) * y_axis_arr[1:-1]
            + (x_axis_arr[:-2] * y_axis_arr[2:] - x_axis_arr[2:] * y_axis_arr[:-2])
        )
        deviation_distance_lower_part = np.sqrt(
            (x_axis_arr[2:] - x_axis_arr[:-2]) ** 2
            + (y_axis_arr[2:] - y_axis_arr[:-2]) ** 2
        )
        deviation_distance = np.zeros_like(deviation_distance_lower_part)
        np.divide(
            np.abs(deviation_distance_upper_part),
            deviation_distance_lower_part,
            out=deviation_distance,
            where=deviation_distance_lower_part != 0,
        )
        self._extracted_features["deviation_distance"] = np.concatenate(
            ([0], deviation_distance, [0])
        )

        # 6. Straightness / Efficiency
        # straightness = np.zeros_like(self._diff_time_arr)
        # np.divide(
        #     self._traveled_distance,
        #     self.curve_length,
        #     out=straightness,
        #     where=self.curve_length != 0,
        # )
        # self._extracted_features["straightness"] = straightness

        # 7. Jitter
        # Ratio of raw step length to smoothed step length.  Values > 1 mean
        # the cursor deviated from the smooth path (tremor / noise); 1 means
        # perfectly smooth.
        #
        # mode="same" zero-pads the signal at both ends, which distorts the
        # first and last (window_size // 2) values.  Instead we use
        # mode="valid" on an asymmetrically edge-padded signal so every output
        # point uses a full, real neighbourhood and the output length is
        # always exactly n, regardless of whether window_size is even or odd.
        #   pad_left  = (w - 1) // 2   →  floor half
        #   pad_right = w // 2          →  ceil  half
        # This satisfies pad_left + pad_right == window_size - 1, which is
        # the exact condition for len(valid output) == len(original signal).
        pad_left  = (window_size - 1) // 2
        pad_right = window_size // 2
        x_padded = np.pad(x_axis_arr, (pad_left, pad_right), mode="edge")
        y_padded = np.pad(y_axis_arr, (pad_left, pad_right), mode="edge")
        kernel = np.ones(window_size) / window_size
        smoothed_x = np.convolve(x_padded, kernel, mode="valid")
        smoothed_y = np.convolve(y_padded, kernel, mode="valid")
        smoothed_path_length = np.sqrt(
            np.concatenate(([0], np.diff(smoothed_x))) ** 2
            + np.concatenate(([0], np.diff(smoothed_y))) ** 2
        )
        jitter = np.zeros_like(self._diff_time_arr)
        np.divide(
            self._traveled_distance,
            smoothed_path_length,
            out=jitter,
            where=smoothed_path_length != 0,
        )
        self._extracted_features["jitter"] = jitter

    def _calculate_speed_features(self) -> None:
        # 8. Speed
        self._speed = np.zeros_like(self._diff_time_arr)
        np.divide(
            self._traveled_distance,
            self._diff_time_arr,
            out=self._speed,
            where=self._diff_time_arr != 0,
        )
        self._extracted_features["speed"] = self._speed

        # 9. Horizontal Speed
        self._horizontal_speed = np.zeros_like(self._diff_time_arr)
        np.divide(
            self._diff_x_axis_arr,
            self._diff_time_arr,
            out=self._horizontal_speed,
            where=self._diff_time_arr != 0,
        )
        self._extracted_features["horizontal_speed"] = self._horizontal_speed

        # 10. Vertical Speed
        self._vertical_speed = np.zeros_like(self._diff_time_arr)
        np.divide(
            self._diff_y_axis_arr,
            self._diff_time_arr,
            out=self._vertical_speed,
            where=self._diff_time_arr != 0,
        )
        self._extracted_features["vertical_speed"] = self._vertical_speed

    def _calculate_acceleration_features(self) -> None:
        diff_x_speed_arr = np.concatenate(([0], np.diff(self._horizontal_speed)))
        diff_y_speed_arr = np.concatenate(([0], np.diff(self._vertical_speed)))
        diff_speed_arr = np.concatenate(([0], np.diff(self._speed)))

        # 11. Horizontal Acceleration
        self._horizontal_acceleration = np.zeros_like(self._diff_time_arr)
        np.divide(
            diff_x_speed_arr,
            self._diff_time_arr,
            out=self._horizontal_acceleration,
            where=self._diff_time_arr != 0,
        )
        self._extracted_features["horizontal_acceleration"] = self._horizontal_acceleration

        # 12. Vertical Acceleration
        self._vertical_acceleration = np.zeros_like(self._diff_time_arr)
        np.divide(
            diff_y_speed_arr,
            self._diff_time_arr,
            out=self._vertical_acceleration,
            where=self._diff_time_arr != 0,
        )
        self._extracted_features["vertical_acceleration"] = self._vertical_acceleration

        # 13. Acceleration
        self._acceleration = np.zeros_like(self._diff_time_arr)
        np.divide(
            diff_speed_arr,
            self._diff_time_arr,
            out=self._acceleration,
            where=self._diff_time_arr != 0,
        )
        self._extracted_features["acceleration"] = self._acceleration

        # 14. Average Speed against distance
        # Computed per window in _extract_statistics to avoid carrying the
        # global cumsum across window boundaries.

        # 15. Horizontal Acceleration vs Resultant
        h_acc_vs_resultant = np.zeros_like(diff_speed_arr)
        np.divide(
            diff_x_speed_arr,
            diff_speed_arr,
            out=h_acc_vs_resultant,
            where=diff_speed_arr != 0,
        )
        self._extracted_features["x_acceleration_vs_resultant"] = h_acc_vs_resultant

        # 16. Vertical Acceleration vs Resultant
        v_acc_vs_resultant = np.zeros_like(diff_speed_arr)
        np.divide(
            diff_y_speed_arr,
            diff_speed_arr,
            out=v_acc_vs_resultant,
            where=diff_speed_arr != 0,
        )
        self._extracted_features["y_acceleration_vs_resultant"] = v_acc_vs_resultant

        # 17. Average X Acceleration against distance
        # Computed per window in _extract_statistics to avoid carrying the
        # global cumsum across window boundaries.

        # 18. Average Y Acceleration against distance
        # Computed per window in _extract_statistics to avoid carrying the
        # global cumsum across window boundaries.

    def _calculate_angle_features(self) -> None:
        # 19. Tangential Speed
        tangential_speed = np.sqrt(
            np.power(self._horizontal_speed, 2) + np.power(self._vertical_speed, 2)
        )
        self._extracted_features["tangential_speed"] = tangential_speed

        # 20. Tangential Acceleration
        tangential_acceleration = np.sqrt(
            np.power(self._horizontal_acceleration, 2)
            + np.power(self._vertical_acceleration, 2)
        )
        self._extracted_features["tangential_acceleration"] = tangential_acceleration

        diff_tang_acc = np.concatenate(([0], np.diff(tangential_acceleration)))

        # 21. Tangential Jerk
        tangential_jerk = np.zeros_like(self._diff_time_arr)
        np.divide(
            diff_tang_acc,
            self._diff_time_arr,
            out=tangential_jerk,
            where=self._diff_time_arr != 0,
        )
        self._extracted_features["tangential_jerk"] = tangential_jerk

        # 22. Angle of movement
        angle = np.arctan2(self._diff_y_axis_arr, self._diff_x_axis_arr)
        self._extracted_features["angle"] = angle

        # 23. Rate of curvature — TODO

        # 24. Total Angles
        # The per-point cumsum is a global accumulator that bleeds across
        # window boundaries.  The angle array (feature 22) is already stored,
        # so _extract_statistics can sum it locally within each window to
        # produce the correct per-window total_angles scalar.

        # 25–27. Regularity / Center of Mass / Scattering Coefficient — TODO

        # 28. Curvature
        # Signed curvature of the trajectory at each point using the
        # Frenet-Serret formula in parametric form:
        #
        #   κ = (vx · ay − vy · ax) / (vx² + vy²)^(3/2)
        #
        # where vx/vy are the horizontal/vertical speed components and
        # ax/ay are the horizontal/vertical acceleration components.
        # The denominator is the cube of the speed magnitude; we guard
        # against division by zero when the cursor is stationary.
        curvature_numerator = (
            self._horizontal_speed * self._vertical_acceleration
            - self._vertical_speed * self._horizontal_acceleration
        )
        curvature_denominator = np.power(
            self._horizontal_speed ** 2 + self._vertical_speed ** 2, 3 / 2
        )
        curvature = np.zeros_like(curvature_denominator)
        np.divide(
            curvature_numerator,
            curvature_denominator,
            out=curvature,
            where=curvature_denominator != 0,
        )
        self._extracted_features["curvature"] = curvature

        # 29–31. Central Moments / Self-Intersection / Angle (cosines) — TODO
        # 32. Acceleration Beginning time — calculated in _extract_statistics
        # 33–34. Skewness / Kurtosis — TODO

    @staticmethod
    def _write_debug_files(user) -> None:
        """
        Write the processed session DataFrames for one user to parquet.

        :param user: UserDataDto whose sessions should be written
        """
        directory_path = Path(f"../datasets/features/user{user.id}")
        directory_path.mkdir(parents=True, exist_ok=True)
        log_dataframe_sessions(directory_path / "training", user.training_sessions)
        log_dataframe_sessions(directory_path / "testing", user.testing_sessions)
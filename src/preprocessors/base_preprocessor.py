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

    # Number of trajectory points grouped into one statistical window.
    WINDOW_SIZE: int = 10

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

    def __init__(self, is_debug: bool = False):
        """
        :param is_debug: When True, write intermediate DataFrames to parquet.
        """
        self.is_debug = is_debug

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
          1. Extract per-point kinematic features (velocity, acceleration, angles…)
          2. Deduplicate consecutive identical (x, y) points
          3. Slice into non-overlapping windows of WINDOW_SIZE
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
        Divide features_df into non-overlapping windows of WINDOW_SIZE rows and
        compute [mean, std, max, min] for every feature column in a single pass.

        :param features_df: Deduplicated per-point feature DataFrame for one session
        :param authentic: Authenticity label (0 or 1) for this session
        :return: Statistics DataFrame with one row per window
        """
        feature_cols = [c for c in features_df.columns if c not in _META_COLS]
        n = len(features_df)
        window_ids = np.arange(n) // self.WINDOW_SIZE

        agg_df = features_df[feature_cols].groupby(window_ids).agg(_STAT_FUNCS)

        agg_df.columns = [f"{stat}_{col}" for col, stat in agg_df.columns]
        agg_df = agg_df.reset_index(drop=True)

        timestamps = features_df["timestamp"].to_numpy()
        window_starts = np.arange(0, n, self.WINDOW_SIZE)
        window_ends = np.minimum(window_starts + 1, n - 1)

        acc_beginning_time = np.where(
            window_starts < window_ends,
            timestamps[window_ends] - timestamps[window_starts],
            0.0,
        )
        agg_df["acc_beginning_time"] = acc_beginning_time[: len(agg_df)]
        agg_df["authentic"] = authentic

        return agg_df

    @property
    def features_dataframe(self) -> pd.DataFrame:
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
        self.curve_length = np.cumsum(self._traveled_distance)
        self._extracted_features["curve_length"] = self.curve_length

        # 3. Elapsed Time / Mouse Digraph
        self._extracted_features["elapsed_time"] = self._diff_time_arr

        # 4. Movement Offset
        self._extracted_features["movement_offset"] = (
            self.curve_length - self._traveled_distance
        )

        # 5. Deviation Distance
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
            deviation_distance_upper_part,
            deviation_distance_lower_part,
            out=deviation_distance,
            where=deviation_distance_lower_part != 0,
        )
        self._extracted_features["deviation_distance"] = np.concatenate(
            ([0], deviation_distance, [0])
        )

        # 6. Straightness / Efficiency
        straightness = np.zeros_like(self._diff_time_arr)
        np.divide(
            self._traveled_distance,
            self.curve_length,
            out=straightness,
            where=self.curve_length != 0,
        )
        self._extracted_features["straightness"] = straightness

        # 7. Jitter
        smoothed_x = np.convolve(
            x_axis_arr, np.ones(window_size) / window_size, mode="same"
        )
        smoothed_y = np.convolve(
            y_axis_arr, np.ones(window_size) / window_size, mode="same"
        )
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
        avg_speed_against_distance = np.zeros_like(self.curve_length)
        np.divide(
            np.cumsum(self._speed) / np.arange(1, len(self._speed) + 1),
            self.curve_length,
            out=avg_speed_against_distance,
            where=self.curve_length != 0,
        )
        self._extracted_features["avg_speed_against_distance"] = avg_speed_against_distance

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
        avg_x_acc_against_distance = np.zeros_like(self.curve_length)
        np.divide(
            np.cumsum(self._horizontal_acceleration)
            / np.arange(1, len(self._horizontal_acceleration) + 1),
            self.curve_length,
            out=avg_x_acc_against_distance,
            where=self.curve_length != 0,
        )
        self._extracted_features["avg_x_acc_against_distance"] = avg_x_acc_against_distance

        # 18. Average Y Acceleration against distance
        avg_y_acc_against_distance = np.zeros_like(self.curve_length)
        np.divide(
            np.cumsum(self._vertical_acceleration)
            / np.arange(1, len(self._vertical_acceleration) + 1),
            self.curve_length,
            out=avg_y_acc_against_distance,
            where=self.curve_length != 0,
        )
        self._extracted_features["avg_y_acc_against_distance"] = avg_y_acc_against_distance

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
        angle = np.zeros_like(self._diff_time_arr)
        np.divide(
            diff_tang_acc,
            self._diff_time_arr,
            out=angle,
            where=self._diff_time_arr != 0,
        )
        self._extracted_features["angle"] = angle

        # 23. Rate of curvature — TODO

        # 24. Total Angles
        self._extracted_features["total_angles"] = np.cumsum(angle)

        # 25–27. Regularity / Center of Mass / Scattering Coefficient — TODO

        # 28. Curvature Velocity
        curvature_velocity = np.zeros_like(tangential_acceleration)
        np.divide(
            tangential_jerk,
            np.power(1 + np.power(tangential_acceleration, 2), 3 / 2),
        )
        self._extracted_features["curvature_velocity"] = curvature_velocity

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
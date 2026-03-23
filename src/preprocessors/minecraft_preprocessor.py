"""
Preprocessing the features following the `Continuous Authentication Using Mouse
Movements, Machine Learning, and Minecraft` article.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.dto import ExtractionData
from src.preprocessors import BasePreprocessor
from src.utils.log_file import log_dataframe_sessions

logger = logging.getLogger(__name__)

# Columns excluded from statistical aggregation.
_META_COLS = frozenset({"x", "y", "timestamp", "authentic", "session"})

# Statistics computed for every feature column in a single .agg() pass.
_STAT_FUNCS = ["mean", "std", "max", "min"]


class MinecraftPreprocessor(BasePreprocessor):
    """
    Extract behavioral biometric features from mouse trajectories following:
    *Continuous Authentication Using Mouse Movements, Machine Learning, and Minecraft*

    Operates on the dict[str, DataFrame] session contract introduced in UserDataDto.
    Each session is processed independently
    """

    # Number of trajectory points grouped into one statistical window.
    WINDOW_SIZE: int = 10

    def preprocess(self, extraction_data: ExtractionData) -> ExtractionData:
        """
        Preprocess all sessions for every user.

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
        Full pipeline for one session:
          1. Extract per-point kinematic features (velocity, acceleration, angles…)
          2. Deduplicate consecutive identical (x, y) points with numpy boolean indexing
          3. Slice into non-overlapping windows of WINDOW_SIZE
          4. Compute [mean, std, max, min] per window in one vectorised .agg() pass

        :param session_df: Raw DataFrame for a single session
        :return: Statistics DataFrame (one row per window)
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

    def _extract_general_features_from_df(self, dataframe: pd.DataFrame) -> None:
        """Not used in the session-based pipeline; kept for ABC compliance."""
        x = dataframe["x"].to_numpy(dtype=float)
        y = dataframe["y"].to_numpy(dtype=float)
        t = dataframe["timestamp"].to_numpy(dtype=float)
        self._calculate_basic_axis_features(x, y, t)
        self._calculate_speed_features()
        self._calculate_acceleration_features()
        self._calculate_angle_features()

    def _extract_statistical_info_from_features_df(self) -> pd.DataFrame:
        """Not used in the session-based pipeline; kept for ABC compliance."""
        raise NotImplementedError(
            "Use _process_single_session() which handles session isolation "
            "and vectorised statistics correctly."
        )

    @staticmethod
    def _write_debug_files(user) -> None:
        directory_path = Path(f"../datasets/features/user{user.id}")
        directory_path.mkdir(parents=True, exist_ok=True)
        log_dataframe_sessions(directory_path / "training", user.training_sessions)
        log_dataframe_sessions(directory_path / "testing", user.testing_sessions)
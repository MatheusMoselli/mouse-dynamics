"""
Preprocessing the features following the `Mouse Dynamics Behavioral Biometrics: A Survey` article.
"""
import numpy as np

from src.dto import ExtractionData, EnumTypeOfSession
from src.preprocessors import BasePreprocessor
from pathlib import Path
import pandas as pd
import logging

from src.utils.log_file import log_dataframe_file, log_dataframe_sessions

logger = logging.getLogger(__name__)

# Columns excluded from statistical aggregation.
_META_COLS = frozenset({"x", "y", "timestamp", "authentic", "session"})

# Statistics computed for every feature column.
_STAT_FUNCS = ["mean", "std", "max", "min"]

class KhanPreprocessor(BasePreprocessor):
    """
    Extract behavioral biometric features from mouse trajectory following the guide
    in the article: *Continuous Authentication Using Mouse Movements, Machine Learning, and Minecraft*
    """

    # Number of trajectory points to group together for feature extraction.
    # The trajectory will be divided into non-overlapping windows of this size.
    WINDOW_SIZE = 10

    def preprocess(self, extraction_data: ExtractionData) -> ExtractionData:
        """
        Preprocess all the dataframes by user

        :param extraction_data: The users standardized dataframes
        :return: Dataframe containing all extracted features with descriptive names
        """
        for user in extraction_data.users:
            logger.info(f"Preprocessing user [TRAINING]: {user.id}")

            user.training_sessions = self._process_sessions(user.training_sessions)
            user.testing_sessions = self._process_sessions(user.testing_sessions)

            logger.info(f"User {user.id} statistical features extracted")

            if self.is_debug:
                self._write_debug_files(user)

        return extraction_data

    def _process_sessions(self, sessions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        result_sessions = { }

        for session_name, session_df in sessions.items():
            if session_df.empty:
                logger.warning(f"[{session_name}] Session is empty, skipping.")
                continue

            stats_df = self._process_single_session(session_df)
            sessions[session_name] = stats_df

        return result_sessions

    def _process_single_session(self, session_name: str, session_df: pd.DataFrame) -> pd.DataFrame:
        authentic = int(session_df["authentic"].iat[0])

        x = session_df["x"].to_numpy(dtype=float)
        y = session_df["y"].to_numpy(dtype=float)
        t = session_df["timestamp"].to_numpy(dtype=float)

        self._initialize_extracted_features_df(
            session_df["authentic"].to_numpy()
        )

        self._calculate_basic_axis_features(x, y, t)
        self._calculate_speed_features()
        self._calculate_acceleration_features()
        self._calculate_angle_features()

        features_df = self.features_dataframe.copy()

        # Deduplicate consecutive identical (x, y) points
        xy = features_df[["x", "y"]].to_numpy()
        is_unique = np.ones(len(xy), dtype=bool)
        is_unique[1:] = (xy[1:] != xy[:-1]).any(axis=1)
        features_df = features_df.loc[is_unique].reset_index(drop=True)

        if features_df.empty:
            return pd.DataFrame()

        return self._extract_statistics(features_df, authentic)

    def _extract_statistics(self, features_df: pd.DataFrame, authentic: int) -> pd.DataFrame:
        feature_cols = [c for c in features_df.columns if c not in _META_COLS]
        n = len(features_df)
        window_ids = np.arange(n) // self.WINDOW_SIZE

        agg_df = (
            features_df[feature_cols]
            .groupby(window_ids)
            .agg(_STAT_FUNCS)
        )

        agg_df.columns = [f"{stat}_{col}" for col, stat in agg_df.columns]
        agg_df = agg_df.reset_index(drop=True)

        timestamps = features_df["timestamp"].to_numpy()
        window_starts = np.arange(0, n, self.WINDOW_SIZE)
        window_ends = np.minimum(window_starts + 1, n - 1)  # index of second point

        acc_beginning_time = np.where(
            window_starts < window_ends,
            timestamps[window_ends] - timestamps[window_starts],
            0.0,
        )

        agg_df["acc_beginning_time"] = acc_beginning_time[: len(agg_df)]
        agg_df["authentic"] = authentic

        return agg_df

    def _write_debug_files(self, user) -> None:
        directory_path = Path(f"../datasets/features/user{user.id}")
        directory_path.mkdir(parents=True, exist_ok=True)

        log_dataframe_sessions(directory_path / "training", user.training_sessions)
        log_dataframe_sessions(directory_path / "testing", user.testing_sessions)
"""
Preprocessing the features following the `Mouse Dynamics Behavioral Biometrics: A Survey` article.
"""
import logging
from src.preprocessors import BasePreprocessor
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class KhanPreprocessor(BasePreprocessor):
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
            general_features_df = self._test(dataframe["x"], dataframe["y"], dataframe["timestamp"], 10)
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
                    and i_df[["x", "y"]].iloc[0].equals(last_value_previous_group)):
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

            columns = [col for col in i_df_clean.columns.tolist() if col not in {"x", "y", "timestamp"}]

            for col_name in columns:
                i_df_statistics[f"mean_{col_name}"] = i_df_clean[col_name].mean()
                i_df_statistics[f"std_{col_name}"] = i_df_clean[col_name].std()
                i_df_statistics[f"max_{col_name}"] = i_df_clean[col_name].max()
                i_df_statistics[f"min_{col_name}"] = i_df_clean[col_name].min()

            # TODO: check if I should do this in the clean df or the original (with duplicates)
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
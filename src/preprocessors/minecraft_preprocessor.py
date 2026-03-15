"""
Preprocessing the features following the `Continuous Authentication Using Mouse Movements,
Machine Learning, and Minecraft` article.
"""
from src.dto import ExtractionData
from src.preprocessors import BasePreprocessor
from pathlib import Path
import pandas as pd
import logging

from src.utils.log_file import log_dataframe_file

logger = logging.getLogger(__name__)

class MinecraftPreprocessor(BasePreprocessor):
    """
    Extract behavioral biometric features from mouse trajectory following the guide
    in the article: *Continuous Authentication Using Mouse Movements, Machine Learning, and Minecraft*
    """

    # Number of trajectory points to group together for feature extraction.
    # The trajectory will be divided into non-overlapping windows of this size.
    AMOUNT_OF_LINES_IN_SEQUENCE = 10

    def preprocess(self,  extraction_data: ExtractionData) -> ExtractionData:
        """
        Preprocess all the dataframes by user

        :param extraction_data: The users standardized dataframes
        :return: Dataframe containing all extracted features with descriptive names
        """
        for user in extraction_data.users:
            logger.info(f"Preprocessing user [TRAINING]: {user.id}")

            session = user.training_dataframe["session"].values
            authentic = user.training_dataframe["authentic"].values
            self._initialize_extracted_features_df(authentic, session)
            self._extract_general_features_from_df(user.training_dataframe)

            training_statistical_df = self._extract_statistical_info_from_features_df()
            user.training_dataframe = training_statistical_df

            logger.info(f"Preprocessing user [TEST]: {user.id}")

            session = user.testing_dataframe["session"].values
            authentic = user.testing_dataframe["authentic"].values
            self._initialize_extracted_features_df(authentic, session)
            self._extract_general_features_from_df(user.testing_dataframe)

            testing_statistical_df = self._extract_statistical_info_from_features_df()
            user.testing_dataframe = testing_statistical_df

            logger.info(f"User {user.id} statistical features extracted")

            if self.is_debug:
                directory_path = Path(f"../datasets/features/user{user.id}")
                directory_path.mkdir(parents=True, exist_ok=True)

                training_path_str = directory_path / "training.parquet"
                testing_path_str = directory_path / "testing.parquet"

                log_dataframe_file(training_path_str, user.training_dataframe)
                log_dataframe_file(testing_path_str, user.testing_dataframe)

        return extraction_data

    def _extract_general_features_from_df(self, dataframe: pd.DataFrame):
        """
        Extract all movement features from a trajectory.

        :return: Dataframe containing all extracted features with descriptive names
        """

        # Convert to numpy arrays for performance
        x = dataframe["x"].values.astype(float)
        y = dataframe["y"].values.astype(float)
        t = dataframe["timestamp"].values.astype(float)

        self._calculate_basic_axis_features(x, y, t)
        self._calculate_speed_features()
        self._calculate_acceleration_features()
        self._calculate_angle_features()

    def _extract_statistical_info_from_features_df(self) -> pd.DataFrame:
        statistics_extracted_arr = []

        for session, df_by_session in self.features_dataframe.groupby("session"):
            grouped_by_df = self.features_dataframe.groupby(self.features_dataframe.index // self.AMOUNT_OF_LINES_IN_SEQUENCE)

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

                i_df_statistics = {}

                columns = [col for col in i_df_clean.columns.tolist() if col not in  {"x", "y", "timestamp", "authentic", "session"}]

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

                i_df_statistics["session"] = session
                i_df_statistics["authentic"] = df_by_session["authentic"].iloc[0]
                statistics_extracted_arr.append(i_df_statistics)

        return pd.DataFrame(statistics_extracted_arr)

"""
Splitting the data following the `Continuous Authentication Using Mouse Movements,
Machine Learning, and Minecraft` Mouse Dynamics dataset.
There will be the same amount of authentic data and unauthentic. Also, the unauthentic data will be populated
by taking the same amount of data from each of the other users (so if there are other 15 users, each will be
responsible for 1/15 of the unauthentic data).
"""
from src.dto import ExtractionData, EnumTypeOfSession, UserDataDto
from src.splitters import BaseSplitter
from pathlib import Path
import pandas as pd

class MinecraftSplitter(BaseSplitter):
    """
    Split the dataset into train/test sets.
    The features should already be extracted at this point
    """

    def split(self, extraction_data: ExtractionData) -> ExtractionData:
        main_users_features = [user for user in extraction_data.users if int(user.id) < 15]
        support_users_features = [user for user in extraction_data.users if int(user.id) >= 15]

        self._split_by_session_type(
            EnumTypeOfSession.TRAINING,
            main_users_features,
            support_users_features,
            extraction_data
        )

        self._split_by_session_type(
            EnumTypeOfSession.TESTING,
            main_users_features,
            support_users_features,
            extraction_data
        )

        return extraction_data

    def _split_by_session_type(self,
                              type_of_session: EnumTypeOfSession,
                              main_users: list[UserDataDto],
                              support_users: list[UserDataDto],
                              extraction_data: ExtractionData):
        for main_user in main_users:
            true_user_df = main_user.training_sessions.copy() \
                if type_of_session == EnumTypeOfSession.TRAINING \
                else main_user.testing_sessions.copy()

            true_user_df["authentic"] = 1

            authentic_df_size = len(true_user_df)
            amount_of_support_users = len(support_users)
            non_authentic_df_sizes = int(authentic_df_size / amount_of_support_users)

            all_dfs = [true_user_df]
            for support_user in support_users:
                if main_user.id == support_user.id:
                    continue

                support_df = support_user.training_sessions.head(non_authentic_df_sizes).copy() \
                    if type_of_session == EnumTypeOfSession.TRAINING \
                    else support_user.testing_sessions.head(non_authentic_df_sizes).copy()

                support_df["authentic"] = 0
                support_df["session"] = "Train" \
                    if type_of_session == EnumTypeOfSession.TRAINING \
                    else "Test"

                all_dfs.append(support_df)

            final_df = pd.concat(all_dfs, ignore_index=True)
            original_user = extraction_data.get_user_by_id(main_user.id)
            if type_of_session == EnumTypeOfSession.TRAINING:
                original_user.training_sessions = final_df
            else:
                original_user.testing_sessions = final_df


            if self.is_debug:
                file_path_str = f"../datasets/training/user{main_user.id}.parquet"

                file = Path(file_path_str)
                file.unlink(missing_ok=True)


                final_df.to_parquet(file_path_str, index=False)


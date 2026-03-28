"""
Splitting the data following the `Continuous Authentication Using Mouse Movements,
Machine Learning, and Minecraft` Mouse Dynamics dataset.
There will be the same amount of authentic data and unauthentic. Also, the unauthentic data will be populated
by taking the same amount of data from each of the other users (so if there are other 15 users, each will be
responsible for 1/15 of the unauthentic data).
"""
from src.dto import ExtractionData, EnumTypeOfSession, UserDataDto
from src.splitters import BaseSplitter
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
            support_users_features
        )

        self._split_by_session_type(
            EnumTypeOfSession.TESTING,
            main_users_features,
            support_users_features
        )

        if self.is_debug:
            for user in extraction_data.users:
                self._write_debug_file(user)

        return extraction_data

    @staticmethod
    def _split_by_session_type(type_of_session: EnumTypeOfSession,
                               main_users: list[UserDataDto],
                               support_users: list[UserDataDto]):
        for main_user in main_users:
            true_user_df = main_user.merged_sessions(type_of_session).copy()

            authentic_df_size = len(true_user_df)
            n_support = len(support_users)
            per_support_size = authentic_df_size // n_support

            all_dfs = [true_user_df]
            for support_user in support_users:
                if main_user.id == support_user.id:
                    continue

                support_df = (
                    support_user.merged_sessions(type_of_session)
                    .head(per_support_size)
                    .copy()
                )
                support_df["authentic"] = 0
                all_dfs.append(support_df)

            final_df = pd.concat(all_dfs, ignore_index=True)

            if type_of_session == EnumTypeOfSession.TRAINING:
                main_user.training_sessions = {"_merged": final_df}
            else:
                main_user.testing_sessions = {"_merged": final_df}
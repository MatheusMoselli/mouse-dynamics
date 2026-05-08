"""
There will be the same amount of authentic data and unauthentic. Also, the unauthentic data will be populated
by taking the same amount of data from each of the other users.
"""
import gc
import logging
from src.dto import ExtractionData, EnumTypeOfSession
from src.splitters import BaseSplitter
import pandas as pd

logger = logging.getLogger(__name__)

class HalfSplitter(BaseSplitter):
    """
    Split the dataset into train/test sets.
    The features should already be extracted at this point
    """

    def split(self, extraction_data: ExtractionData) -> ExtractionData:
        """
        Split the dataset into train/test sets.
        :param extraction_data:  the list of datasets to be split
        :return: The list of train/test sets
        """
        users = extraction_data.users
        n_support = len(users) - 1

        training_cache: dict[str, pd.DataFrame] = {}
        for user in users:
            training_cache[user.id] = user.merged_sessions(EnumTypeOfSession.TRAINING)
            user.training_sessions = {}

        gc.collect()

        for user in users:
            true_user_df = training_cache[user.id].copy()

            authentic_df_size = len(true_user_df)
            per_support_size = authentic_df_size // n_support

            all_training_dfs = [true_user_df]

            for support_user in users:
                if support_user.id == user.id:
                    continue

                support_df = training_cache[support_user.id].iloc[:per_support_size].copy()
                support_df["authentic"] = 0
                all_training_dfs.append(support_df)

            final_training_df = pd.concat(all_training_dfs, ignore_index=True)
            del all_training_dfs

            merged_testing_df = user.merged_sessions(EnumTypeOfSession.TESTING)
            user.training_sessions = {"_merged": final_training_df}
            user.testing_sessions  = {"_merged": merged_testing_df}

            if self.is_debug:
                self._write_debug_file(user)

        training_cache.clear()
        gc.collect()

        return extraction_data
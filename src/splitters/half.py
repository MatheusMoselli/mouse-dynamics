"""
There will be the same amount of authentic data and unauthentic. Also, the unauthentic data will be populated
by taking the same amount of data from each of the other users.
"""
import logging
from src.dto import ExtractionData, EnumTypeOfSession
from src.splitters import BaseSplitter
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class HalfSplitter(BaseSplitter):
    """
    Split the dataset into train/test sets.
    The features should already be extracted at this point
    """

    def split(self, extraction_data: ExtractionData) -> ExtractionData:
        for user in extraction_data.users:
            true_user_df = user.merged_sessions(EnumTypeOfSession.TRAINING).copy()

            authentic_df_size = len(true_user_df)
            n_support = len(extraction_data.users) - 1
            per_support_size = authentic_df_size // n_support

            all_training_dfs = [true_user_df]
            for support_user in extraction_data.users:
                if user.id == support_user.id:
                    continue

                support_df = (
                    support_user.merged_sessions(EnumTypeOfSession.TRAINING)
                    .head(per_support_size)
                    .copy()
                )

                support_df["authentic"] = 0
                all_training_dfs.append(support_df)

            final_training_df = pd.concat(all_training_dfs, ignore_index=True)
            user.training_sessions = final_training_df
            user.training_sessions = { "_merged": final_training_df }

            merged_testing_df = user.merged_sessions(EnumTypeOfSession.TESTING)
            user.testing_sessions = { "_merged": merged_testing_df }

            if self.is_debug:
                self._write_debug_file(user)

        return extraction_data

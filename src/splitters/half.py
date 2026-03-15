"""
There will be the same amount of authentic data and unauthentic. Also, the unauthentic data will be populated
by taking the same amount of data from each of the other users.
"""
import logging
from src.dto import ExtractionData
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
            true_user_training_df = user.training_dataframe.copy()

            authentic_df_size = len(true_user_training_df)
            amount_of_support_users = len(extraction_data.users) - 1 # Excluding current user
            non_authentic_df_sizes = int(authentic_df_size / amount_of_support_users)

            all_training_dfs = [true_user_training_df]
            for support_user in extraction_data.users:
                if user.id == support_user.id:
                    continue

                ith_false_user_training_df = support_user.training_dataframe.head(non_authentic_df_sizes).copy()
                ith_false_user_training_df["authentic"] = 0
                all_training_dfs.append(ith_false_user_training_df)

            final_training_df = pd.concat(all_training_dfs, ignore_index=True)
            user.training_dataframe = final_training_df

            if self.is_debug:
                try:
                    file_path_str = f"../datasets/training/user{user.id}.parquet"

                    file = Path(file_path_str)
                    file.unlink(missing_ok=True)


                    final_training_df.to_parquet(file_path_str, index=False)
                except OSError as e:
                    # If the directory is not found, the code is running locally, so needs to go one layer up.
                    file_path_str = f"../../datasets/training/user{user.id}.parquet"

                    file = Path(file_path_str)
                    file.unlink(missing_ok=True)

                    logger.error(e)

                    final_training_df.to_parquet(file_path_str, index=False)

        return extraction_data
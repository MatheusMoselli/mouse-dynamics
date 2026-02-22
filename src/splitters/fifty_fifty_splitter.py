"""
There will be the same amount of authentic data and unauthentic. Also, the unauthentic data will be populated
by taking the same amount of data from each of the other users.
"""
from src.splitters import BaseSplitter
from typing import Dict
from pathlib import Path
import pandas as pd

class FiftyFiftySplitter(BaseSplitter):
    """
    Split the dataset into train/test sets.
    The features should already be extracted at this point
    """

    def split(self, dataframes_by_users: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        split_dfs_by_users = {}

        for main_user_id, main_user_df in dataframes_by_users.items():
            true_user_training_df = main_user_df.copy()
            true_user_training_df["authentic"] = 1

            authentic_df_size = len(true_user_training_df)
            amount_of_support_users = len(dataframes_by_users) - 1 # Excluding current user
            non_authentic_df_sizes = int(authentic_df_size / amount_of_support_users)

            all_training_dfs = [true_user_training_df]
            for support_user_id, support_user_df in dataframes_by_users.items():
                if main_user_id == support_user_id:
                    continue

                ith_false_user_training_df = support_user_df.head(non_authentic_df_sizes).copy()
                ith_false_user_training_df["authentic"] = 0
                all_training_dfs.append(ith_false_user_training_df)

            final_training_df = pd.concat(all_training_dfs, ignore_index=True)
            split_dfs_by_users[main_user_id] = final_training_df

            if self.is_debug:
                try:
                    file_path_str = f"../datasets/training/user{main_user_id}.parquet"

                    file = Path(file_path_str)
                    file.unlink(missing_ok=True)


                    final_training_df.to_parquet(file_path_str, index=False)
                except OSError as e:
                    # If the directory is not found, the code is running locally, so needs to go one layer up.
                    file_path_str = f"../../datasets/training/user{main_user_id}.parquet"

                    file = Path(file_path_str)
                    file.unlink(missing_ok=True)


                    final_training_df.to_parquet(file_path_str, index=False)


        return split_dfs_by_users



if __name__ == "__main__":
    fifty_fifty_splitter = FiftyFiftySplitter(is_debug=True)

    feature_files_location = Path("../../datasets/features")
    dfs_by_users = {}
    for feature_file in feature_files_location.iterdir():
        if feature_file.suffix != ".parquet":
            continue

        user_id = feature_file.stem.replace("user","")
        dfs_by_users[user_id] = pd.read_parquet(feature_file)

    fifty_fifty_splitter.split(dfs_by_users)
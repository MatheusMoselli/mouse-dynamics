"""
Implementation for loading the balabit dataset. For it to work properly, you should put the hole dataset
in the following path: mouse-dynamics(root)/raw/balabit/ORIGINAL_DATASET_HERE
"""
from src.dataset_loaders import BaseDatasetLoader
from typing import Dict
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BalabitLoader(BaseDatasetLoader):
    """Loader for `Balabit` Mouse Dynamics dataset."""

    def __init__(self,  is_debug: bool = False):
        """
        Initialize the dataset loader.
        """
        super().__init__(is_debug)
        self.data_path = Path("../datasets/raw/balabit/training_files")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load Balabit dataset.

        Expected structure:
            datasets/raw/balabit/
            ├── test_files/
            ├──── user1/
            ├────── session_0 (without extension)
            ├────── session_1
            ├────── ...
            ├──── user2/
            ├──── ...
            └── training_files/


        CSV format: record timestamp, client timestamp, button, state, x, y
        :return: Dictionary mapping user_id to DataFrame with standardized columns.
        """
        dataframes_by_users = {}

        for directory in self.data_path.iterdir():
            user_id = directory.stem.replace("user","")

            if not directory.is_dir():
                continue

            for session in directory.iterdir():
                session_df = pd.read_csv(session)

                standardized_session_df = self._standardize_columns(
                    session_df,
                    x_col_name="x",
                    y_col_name="y",
                    time_col_name="record timestamp",
                    action_col_name="button",
                )

                if user_id in dataframes_by_users:
                    dataframes_by_users[user_id] = pd.concat([dataframes_by_users[user_id], standardized_session_df])
                else:
                    dataframes_by_users[user_id] = standardized_session_df

            for user_id, dataframe in dataframes_by_users.items():
                if self.is_debug:
                    file_path_str = f"../datasets/base/user{user_id}.parquet"

                    file = Path(file_path_str)
                    file.unlink(missing_ok=True)

                    dataframe.to_parquet(file_path_str, index=False)

        logger.info(f"Loaded {len(dataframes_by_users)} users from Balabit dataset")
        return dataframes_by_users

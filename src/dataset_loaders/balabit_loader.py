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

    def __init__(self):
        """
        Initialize the dataset loader.
        """
        super().__init__()
        self.data_path = Path("../../datasets/raw/balabit/training_files")

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

        for csv_file in sorted(self.data_path.glob("*.csv")):
            logger.info(f"Loading {csv_file.name}")

            df = pd.read_csv(csv_file)

            # Group by user
            for user_id, user_df in df.groupby('user_id'):
                # Standardize columns
                standardized = self._standardize_columns(
                    user_df,
                    x_col_name='x',
                    y_col_name='y',
                    time_col_name='timestamp',
                    action_col_name='state'
                )

                if user_id in dataframes_by_users:
                    dataframes_by_users[str(user_id)] = pd.concat([dataframes_by_users[str(user_id)], standardized])
                else:
                    dataframes_by_users[str(user_id)] = standardized

        logger.info(f"Loaded {len(dataframes_by_users)} users from Balabit dataset")
        return dataframes_by_users

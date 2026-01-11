"""
Implementation for loading the Continuous Authentication Using Mouse Movements, Machine Learning, and Minecraft`
Mouse Dynamics dataset. For it to work properly, you should put the hole dataset
in the following path: mouse-dynamics(root)/raw/minecraft/ORIGINAL_DATASET_HERE
"""
from src.dataset_loaders import BaseDatasetLoader
from pathlib import Path
from typing import Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MinecraftLoader(BaseDatasetLoader):
    """Loader for `Continuous Authentication Using Mouse Movements, Machine Learning, and Minecraft` Mouse Dynamics dataset."""
    def __init__(self, is_debug: bool = False):
        """
        Initialize the data loader.
        """

        super().__init__(is_debug)
        self.data_path = Path("../datasets/raw/minecraft")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load Minecraft dataset.

        Expected structure:
        datasets/raw/minecraft/
            ├── 10extracted (we`ll only use this)
            ├──── Subject0_raw.csv (we`ll only use the raw files)
            ├──── Subject0_test_Extracted.csv
            ├──── Subject0_train_Extracted.csv
            ├──── Subject1_raw.csv
            ├──── Subject1_test_Extracted.csv
            ├──── Subject1_train_Extracted.csv
            ├──── ...
            ├── 10raw
            ├── 40extracted
            └── 40raw

        :return: Dictionary mapping user_id to DataFrame with standardized columns.
        """
        dataframes_by_users = {}

        filename = "masterTrain"
        self.data_path = self.data_path / "40raw" / "masterTrain.csv"
        df = pd.read_csv(self.data_path)

        logger.info(f"Loading {filename}")

        # Standardize columns
        standardized = self._standardize_columns(
            df,
            x_col_name='X',
            y_col_name='Y',
            time_col_name='Timestamp',
            user_id_col_name="Subject ID",
            action_col_name='Button Pressed',
        )

        logger.info(f"{filename} standardized")

        for user_id, user_df in standardized.groupby("user_id"):
            if user_id in dataframes_by_users:
                dataframes_by_users[str(user_id)] = pd.concat([dataframes_by_users[str(user_id)], user_df])
            else:
                dataframes_by_users[str(user_id)] = user_df

        logger.info(f"Loaded {len(dataframes_by_users)} users from Minecraft dataset")
        return dataframes_by_users


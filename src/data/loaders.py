"""
Data loaders for different mouse dynamics datasets.
Handles various formats and structures of behavioral biometric data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal
from abc import ABC, abstractmethod
import json
import logging

from src.data import DatasetsNames

logger = logging.getLogger(__name__)

class BaseDataLoader(ABC):
    """Abstract base class for dataset loaders."""
    @abstractmethod
    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load the dataset.

        :return: Dictionary mapping user_id to DataFrame with standardized columns.
        """
        pass

    @staticmethod
    def _standardize_columns(df: pd.DataFrame,
                             x_col_name: str, y_col_name: str, time_col_name: str,
                             user_id_col_name: Optional[str] = None,
                             action_col_name: Optional[str] = None) -> pd.DataFrame:
        """Standardize column names to [x, y, timestamp, action, session_id]."""
        renamed = df.rename(columns={
            x_col_name: 'x',
            y_col_name: 'y',
            time_col_name: 'timestamp'
        })

        if action_col_name and action_col_name in df.columns:
            renamed = renamed.rename(columns={action_col_name: 'action'})
        else:
            renamed['action'] = 'move'  # Default action

        # Ensure required columns exist and are properly typed
        renamed['x'] = renamed['x'].astype(float)
        renamed['y'] = renamed['y'].astype(float)
        renamed['timestamp'] = pd.to_numeric(renamed['timestamp'], errors='coerce')

        if user_id_col_name and user_id_col_name in df.columns:
            renamed = renamed.rename(columns={user_id_col_name: 'user_id'})
            return renamed[['x', 'y', 'timestamp', 'action', 'user_id']]

        return renamed[['x', 'y', 'timestamp', 'action']]

    #TODO: Add an generic function to standardize the action column, override it in every class.

class BalabitLoader(BaseDataLoader):
    """Loader for `Balabit` Mouse Dynamics dataset."""

    def __init__(self):
        """
        Initialize the data loader.
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
                    dataframes_by_users[user_id] = pd.concat([dataframes_by_users[user_id], standardized])
                else:
                    dataframes_by_users[user_id] = standardized

        logger.info(f"Loaded {len(dataframes_by_users)} users from Balabit dataset")
        return dataframes_by_users

class MinecraftLoader(BaseDataLoader):
    """Loader for `Continuous Authentication Using Mouse Movements, Machine Learning, and Minecraft` Mouse Dynamics dataset."""
    def __init__(self):
        """
        Initialize the data loader.
        """

        super().__init__()
        self.data_path = Path("../../datasets/raw/minecraft")

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

        # Entering the 10extracted folder
        self.data_path = self.data_path / "40raw"

        # for csv_file in sorted(self.data_path.glob("*.csv")):
        #     # Only use the SubjectX_raw.csv files
        #     if not "raw" in csv_file.stem:
        #         continue
        #
        #     logger.info(f"Loading {csv_file.name}")
        #
        #     df = pd.read_csv(csv_file)
        #
        #     user_id = csv_file.stem.replace("Subject", "").split("_")[0]
        #
        #     # Standardize columns
        #     standardized = self._standardize_columns(
        #         df,
        #         x_col_name='X',
        #         y_col_name='Y',
        #         time_col_name='Timestamp',
        #         action_col_name='Button Pressed'
        #     )
        #
        #     if user_id in dataframes_by_users:
        #         dataframes_by_users[user_id] = pd.concat([dataframes_by_users[user_id], standardized])
        #     else:
        #         dataframes_by_users[user_id] = standardized

        # Only use the SubjectX_raw.csv files

        filename = "masterTrain"
        df = pd.read_csv(self.data_path / "masterTrain.csv")

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
                dataframes_by_users[user_id] = pd.concat([dataframes_by_users[user_id], user_df])
            else:
                dataframes_by_users[user_id] = user_df

        logger.info(f"Loaded {len(dataframes_by_users)} users from Minecraft dataset")
        return dataframes_by_users


def load_dataset(dataset_name: DatasetsNames) -> Dict[str, pd.DataFrame]:
    """
    Factory function to load datasets by name.

    :param dataset_name: Name of the dataset
    :return: Dictionary mapping user_id to DataFrame
    """
    loaders = {
        DatasetsNames.BALABIT: BalabitLoader,
        DatasetsNames.MINECRAFT: MinecraftLoader,
    }

    if dataset_name in loaders:
        loader = loaders[dataset_name]()
        return loader.load()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
"""
Generic loader and standardizer for dataset files them for consistent downstream use.
"""
from typing import Dict, Optional
from abc import ABC, abstractmethod
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, is_debug: bool = False):
        """
        Class initialization.
        :param is_debug: Is the dataset loader being run in debug mode.
        """
        self.is_debug = is_debug

    @abstractmethod
    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load the dataset.

        :return: Dictionary mapping user_id to DataFrame with standardized columns.
        """
        pass

    #TODO: check if this is really static
    @staticmethod
    def _standardize_columns(df: pd.DataFrame,
                             x_col_name: str, y_col_name: str, time_col_name: str,
                             user_id_col_name: Optional[str] = None,
                             action_col_name: Optional[str] = None) -> pd.DataFrame:
        """Standardize column names to [x, y, timestamp, action]."""
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

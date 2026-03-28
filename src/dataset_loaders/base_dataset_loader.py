"""
Generic loader and standardizer for dataset files for consistent downstream use.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd
import logging
from src.dto import ExtractionData, UserDataDto, EnumTypeOfSession
from src.utils.log_file import log_dataframe_sessions

logger = logging.getLogger(__name__)

class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, data_path: Path, is_debug: bool = False):
        """
        Initialize the loader, validate the data path, and prepare shared state.

        :param data_path: Root directory of the dataset on disk
        :param is_debug: When True, write intermediate DataFrames to parquet
        """
        self.is_debug = is_debug
        self.data_path = data_path
        self._extraction_data: ExtractionData = ExtractionData()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    @abstractmethod
    def load(self) -> ExtractionData:
        """
        Load the dataset.

        :return: ExtractionData with one UserDataDto per subject.
        """
        pass

    def _get_or_create_user(self, user_id: str) -> UserDataDto:
        """
        Return the existing UserDataDto for user_id, or create and register
        a new one if it does not exist yet.

        :param user_id: String identifier for the user
        :return: The UserDataDto for this user_id
        """
        user = self._extraction_data.get_user_by_id(user_id)
        if user is None:
            user = self._extraction_data.add_user(UserDataDto(user_id))

        return user

    def _write_debug_files(self, debug_subdir: str = "base") -> None:
        """
        Write each user's raw session DataFrames to parquet for inspection.
        :param debug_subdir: Subdirectory under ../datasets/ (default: "base")
        """
        for user in self._extraction_data.users:
            directory_path = Path(f"../datasets/{debug_subdir}/user{user.id}")
            directory_path.mkdir(parents=True, exist_ok=True)

            log_dataframe_sessions(directory_path / "training", user.training_sessions)
            log_dataframe_sessions(directory_path / "testing", user.testing_sessions)

    @staticmethod
    def _standardize_columns(
        df: pd.DataFrame,
        x_col_name: str,
        y_col_name: str,
        time_col_name: str,
        user_id_col_name: Optional[str] = None,
        action_col_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Standardize column names to [x, y, timestamp, action(, user_id)]."""
        renamed = df.rename(
            columns={
                x_col_name: "x",
                y_col_name: "y",
                time_col_name: "timestamp",
            }
        )

        if action_col_name and action_col_name in df.columns:
            renamed = renamed.rename(columns={action_col_name: "action"})
        else:
            renamed["action"] = "move"

        renamed["x"] = renamed["x"].astype(float)
        renamed["y"] = renamed["y"].astype(float)
        renamed["timestamp"] = pd.to_numeric(renamed["timestamp"], errors="coerce")

        if user_id_col_name and user_id_col_name in df.columns:
            renamed = renamed.rename(columns={user_id_col_name: "user_id"})
            return renamed[["x", "y", "timestamp", "action", "user_id"]]

        return renamed[["x", "y", "timestamp", "action"]]

        # TODO: Add a generic method to standardize the action column, overridden in each concrete loader.
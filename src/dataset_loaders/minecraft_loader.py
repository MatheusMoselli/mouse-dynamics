"""
Loader for the `Continuous Authentication Using Mouse Movements, Machine Learning,
and Minecraft` Mouse Dynamics dataset.

Dataset path: mouse-dynamics(root)/datasets/raw/minecraft/
"""
from src.dataset_loaders import BaseDatasetLoader
from src.dto import ExtractionData, EnumTypeOfSession
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# TODO: add support to test files


class MinecraftLoader(BaseDatasetLoader):
    """Loader for the Minecraft Mouse Dynamics dataset."""

    def __init__(self, is_debug: bool = False):
        super().__init__(
            data_path=Path("../datasets/raw/minecraft"),
            is_debug=is_debug,
        )

    def load(self) -> ExtractionData:
        """
        Load the Minecraft dataset.

        Expected structure:
            datasets/raw/minecraft/
            ├── 40raw/
            │   ├── masterTrain.csv
            │   └── masterTest.csv
            └── ...

        CSV columns: Timestamp, X, Y, Button Pressed, Time, DistanceX, DistanceY,
                     Speed, Acceleration, Sex, Subject ID

        :return: ExtractionData populated with one UserDataDto per subject.
        """
        self._load_users(self.data_path / "40raw" / "masterTrain.csv", EnumTypeOfSession.TRAINING)
        self._load_users(self.data_path / "40raw" / "masterTest.csv", EnumTypeOfSession.TESTING)

        if self.is_debug:
            self._write_debug_files()

        logger.info(f"Loaded {len(self._extraction_data.users)} users from Minecraft dataset")
        return self._extraction_data

    def _load_users(self, base_path: Path, type_of_session: EnumTypeOfSession) -> None:
        logger.info(f"Loading {base_path}")
        df = pd.read_csv(base_path)

        standardized = self._standardize_columns(
            df,
            x_col_name="X",
            y_col_name="Y",
            time_col_name="Timestamp",
            user_id_col_name="Subject ID",
            action_col_name="Button Pressed",
        )

        session_name = (
            "Train" if type_of_session == EnumTypeOfSession.TRAINING else "Test"
        )

        for user_id, user_df in standardized.groupby("user_id"):
            user_df = user_df.copy()
            user_df["authentic"] = 1

            user_data = self._get_or_create_user(str(user_id))
            user_data.append_session(session_name, user_df, type_of_session)
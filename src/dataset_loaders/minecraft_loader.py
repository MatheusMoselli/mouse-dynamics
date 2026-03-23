"""
Implementation for loading the `Continuous Authentication Using Mouse Movements,
Machine Learning, and Minecraft` Mouse Dynamics dataset. For it to work properly,
you should put the dataset in: mouse-dynamics(root)/datasets/raw/minecraft/
"""
from src.dataset_loaders import BaseDatasetLoader
from src.dto import ExtractionData, EnumTypeOfSession, UserDataDto
from src.utils.log_file import log_dataframe_sessions
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MinecraftLoader(BaseDatasetLoader):
    """
    Loader for the `Continuous Authentication Using Mouse Movements,
    Machine Learning, and Minecraft` Mouse Dynamics dataset.
    """

    def __init__(self, is_debug: bool = False):
        super().__init__(is_debug)
        self.data_path = Path("../datasets/raw/minecraft")
        self._extraction_data: ExtractionData = ExtractionData()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def load(self) -> ExtractionData:
        """
        Load the Minecraft dataset.

        Expected structure:
            datasets/raw/minecraft/
            ├── 10extracted/
            ├── 10raw/
            │   ├── masterTest.csv
            │   ├── masterTrain.csv
            │   └── masterVal.csv
            ├── 40extracted/
            └── 40raw/
                ├── masterTest.csv
                └── masterTrain.csv

        CSV columns: Timestamp, X, Y, Button Pressed, Time, DistanceX, DistanceY,
                     Speed, Acceleration, Sex, Subject ID

        :return: ExtractionData populated with one UserDataDto per subject.
        """
        train_path = self.data_path / "40raw" / "masterTrain.csv"
        test_path = self.data_path / "40raw" / "masterTest.csv"

        self._load_users(train_path, EnumTypeOfSession.TRAINING)
        self._load_users(test_path, EnumTypeOfSession.TESTING)

        if self.is_debug:
            self._write_debug_files()

        logger.info(f"Loaded {len(self._extraction_data.users)} users from Minecraft dataset")
        return self._extraction_data

    def _load_users(self, base_path: Path, type_of_session: EnumTypeOfSession) -> None:
        """
        Read a master CSV, split by Subject ID, and store each user's rows
        as a named session in their UserDataDto.
        """
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

            user_data = self._extraction_data.get_user_by_id(str(user_id))
            if user_data is None:
                user_data = self._extraction_data.add_user(UserDataDto(str(user_id)))

            user_data.append_session(session_name, user_df, type_of_session)

    def _write_debug_files(self) -> None:
        for user in self._extraction_data.users:
            directory_path = Path(f"../datasets/base/user{user.id}")
            directory_path.mkdir(parents=True, exist_ok=True)

            log_dataframe_sessions(directory_path / "training", user.training_sessions)
            log_dataframe_sessions(directory_path / "testing", user.testing_sessions)
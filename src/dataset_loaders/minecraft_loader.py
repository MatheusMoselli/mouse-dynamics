"""
Implementation for loading the Continuous Authentication Using Mouse Movements, Machine Learning, and Minecraft`
Mouse Dynamics dataset. For it to work properly, you should put the hole dataset
in the following path: mouse-dynamics(root)/raw/minecraft/ORIGINAL_DATASET_HERE
"""
from src.dataset_loaders import BaseDatasetLoader
from src.dto import ExtractionData, EnumTypeOfSession, UserDataDto
from pathlib import Path
import pandas as pd
import logging

from src.utils.log_file import log_dataframe_file

logger = logging.getLogger(__name__)

#TODO: add support to test files

class MinecraftLoader(BaseDatasetLoader):
    """Loader for `Continuous Authentication Using Mouse Movements, Machine Learning, and Minecraft` Mouse Dynamics dataset."""
    def __init__(self, is_debug: bool = False):
        """
        Initialize the data loader.
        """
        super().__init__(is_debug)
        self.data_path = Path("../datasets/raw/minecraft")
        self._extraction_data: ExtractionData = ExtractionData()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def load(self) -> ExtractionData:
        """
        Load Minecraft dataset.

        Expected structure:
        datasets/raw/minecraft/
            ├── 10extracted
            ├──── Subject0_raw.csv
            ├──── Subject0_test_Extracted.csv
            ├──── Subject0_train_Extracted.csv
            ├──── Subject1_raw.csv
            ├──── Subject1_test_Extracted.csv
            ├──── Subject1_train_Extracted.csv
            ├──── ...
            ├── 10raw (we`ll only use this)
            ├──── masterTest.csv
            ├──── masterTrain.csv
            ├──── masterVal.csv
            ├── 40extracted
            └── 40raw

        CSV format: Timestamp, X, Y, Button Pressed, Time, DistanceX, DistanceY, Speed, Acceleration, Sex, Subject ID
        :return: Dictionary mapping user_id to DataFrame with standardized columns.
        """
        train_path = self.data_path / "40raw" / "masterTrain.csv"
        test_path = self.data_path / "40raw" / "masterTest.csv"

        self._load_users(train_path, EnumTypeOfSession.TRAINING)
        self._load_users(test_path, EnumTypeOfSession.TESTING)

        if self.is_debug:
            for user in self._extraction_data.users:
                directory_path = Path(f"../datasets/base/user{user.id}")
                directory_path.mkdir(parents=True, exist_ok=True)

                training_path_str = directory_path / "training.parquet"
                testing_path_str = directory_path / "testing.parquet"

                log_dataframe_file(training_path_str, user.training_dataframe)
                log_dataframe_file(testing_path_str, user.testing_dataframe)

        logger.info(f"Loaded {len(self._extraction_data.users)} users from Minecraft dataset")
        return self._extraction_data

    def _load_users(self, base_path: Path, type_of_session: EnumTypeOfSession):
        df = pd.read_csv(base_path)

        logger.info(f"Loading {base_path}")

        # Standardize columns
        standardized = self._standardize_columns(
            df,
            x_col_name='X',
            y_col_name='Y',
            time_col_name='Timestamp',
            user_id_col_name="Subject ID",
            action_col_name='Button Pressed',
        )

        standardized["session"] = "Train" if type_of_session == EnumTypeOfSession.TRAINING else "Test"

        for user_id, user_df in standardized.groupby("user_id"):
            user_df["authentic"] = 1
            user_data = self._extraction_data.get_user_by_id(str(user_id))

            if user_data is None:
                user_data = self._extraction_data.add_user(UserDataDto(str(user_id)))

            if type_of_session == EnumTypeOfSession.TRAINING:
                user_data.append_dataframe(user_df, EnumTypeOfSession.TRAINING)
            else:
                user_data.append_dataframe(user_df, EnumTypeOfSession.TESTING)


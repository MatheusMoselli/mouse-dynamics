"""
Implementation for loading the balabit dataset. For it to work properly, you should put the hole dataset
in the following path: mouse-dynamics(root)/raw/balabit/ORIGINAL_DATASET_HERE
"""
from src.dataset_loaders import BaseDatasetLoader
from src.utils.log_file import log_dataframe_file
from src.dto import ExtractionData, UserDataDto, EnumTypeOfSession
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
        self.data_path = Path("../datasets/raw/balabit")
        self._extraction_data: ExtractionData = ExtractionData()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def load(self) -> ExtractionData:
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
        self._load_users(self.data_path / "training_files", EnumTypeOfSession.TRAINING)
        self._load_users(self.data_path / "test_files", EnumTypeOfSession.TESTING)

        if self.is_debug:
            for user in self._extraction_data.users:
                directory_path = Path(f"../datasets/base/user{user.id}")
                directory_path.mkdir(parents=True, exist_ok=True)

                training_path_str = directory_path / "training.parquet"
                testing_path_str = directory_path / "testing.parquet"

                log_dataframe_file(training_path_str, user.training_dataframe)
                log_dataframe_file(testing_path_str, user.testing_dataframe)

        logger.info(f"Loaded {len(self._extraction_data.users)} users from Balabit dataset")
        return self._extraction_data

    def _load_users(self, base_path: Path, type_of_session: EnumTypeOfSession):
        for directory in base_path.iterdir():
            if not directory.is_dir():
                continue

            user_id = directory.stem.replace("user","")
            user_data = self._extraction_data.get_user_by_id(user_id)
            if user_data is None:
                user_data = self._extraction_data.add_user(UserDataDto(user_id))

            self._aggregate_sessions_into_file(directory, user_data, type_of_session)


    def _aggregate_sessions_into_file(
            self,
            sessions_directory: Path,
            user_data: UserDataDto,
            type_of_session: EnumTypeOfSession):
        for session in sessions_directory.iterdir():
            session_df = pd.read_csv(session)

            standardized_session_df = self._standardize_columns(
                session_df,
                x_col_name="x",
                y_col_name="y",
                time_col_name="record timestamp",
                action_col_name="button",
            )

            standardized_session_df["session"] = session.stem

            if type_of_session == EnumTypeOfSession.TRAINING:
                standardized_session_df["authentic"] = 1
                user_data.append_dataframe(standardized_session_df, EnumTypeOfSession.TRAINING)
            else:
                authenticity_labels = pd.read_csv(self.data_path / "public_labels.csv")

                standardized_session_df["authentic"] = (standardized_session_df["session"]
                    .map(authenticity_labels.set_index("filename")["is_illegal"])
                    .dropna()
                    #.fillna(0)
                    .astype(int)
                )

                user_data.append_dataframe(standardized_session_df, EnumTypeOfSession.TESTING)

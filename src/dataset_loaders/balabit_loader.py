"""
Loader for the Balabit Mouse Dynamics dataset.

Dataset path: mouse-dynamics(root)/datasets/raw/balabit/
"""
from src.dataset_loaders import BaseDatasetLoader
from src.dto import ExtractionData, UserDataDto, EnumTypeOfSession
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BalabitLoader(BaseDatasetLoader):
    """Loader for the Balabit Mouse Dynamics dataset."""

    def __init__(self, is_debug: bool = False):
        super().__init__(
            data_path=Path("../datasets/raw/balabit"),
            is_debug=is_debug,
        )

        self._authenticity_labels: pd.Series | None = None

    def load(self) -> ExtractionData:
        """
        Load Balabit dataset.

        Expected structure:
            datasets/raw/balabit/
            ├── test_files/
            │   ├── user1/
            │   │   ├── session_0  (no extension)
            │   │   └── session_1
            │   └── user2/
            └── training_files/

        CSV format: record timestamp, client timestamp, button, state, x, y
        """
        labels_path = self.data_path / "public_labels.csv"
        if labels_path.exists():
            self._authenticity_labels = (
                pd.read_csv(labels_path).set_index("filename")["is_illegal"]
            )

        self._load_users(self.data_path / "training_files", EnumTypeOfSession.TRAINING)
        self._load_users(self.data_path / "test_files", EnumTypeOfSession.TESTING)

        if self.is_debug:
            self._write_debug_files()

        logger.info(f"Loaded {len(self._extraction_data.users)} users from Balabit dataset")
        return self._extraction_data

    def _load_users(self, base_path: Path, type_of_session: EnumTypeOfSession) -> None:
        """
        load all users into sessions
        :param base_path: the base path of the dataset
        :param type_of_session: type of session to load
        """
        for directory in base_path.iterdir():
            if not directory.is_dir():
                continue

            user_id = directory.stem.replace("user", "")
            user_data = self._get_or_create_user(user_id)
            self._load_sessions(directory, user_data, type_of_session)

    def _load_sessions(
        self,
        sessions_directory: Path,
        user_data: UserDataDto,
        type_of_session: EnumTypeOfSession,
    ) -> None:
        """
        Load all sessions of the user
        :param sessions_directory: the base path of the user
        :param user_data: the user to load into
        :param type_of_session: type of session to load
        """
        for session_path in sessions_directory.iterdir():
            session_df = pd.read_csv(session_path)

            standardized_df = self._standardize_columns(
                session_df,
                x_col_name="x",
                y_col_name="y",
                time_col_name="record timestamp",
                action_col_name="button",
            )

            session_name = session_path.stem

            if type_of_session == EnumTypeOfSession.TRAINING:
                standardized_df["authentic"] = 1
            else:
                if self._authenticity_labels is None:
                    raise RuntimeError( "public_labels.csv not found; cannot label test sessions.")
                label = self._authenticity_labels.get(session_name)
                if label is None:
                    logger.warning(f"No label found for session {session_name!r}, skipping.")
                    continue
                standardized_df["authentic"] = 1 if int(label) == 0 else 0 # if label = 1, illegal

            user_data.append_session(session_name, standardized_df, type_of_session)
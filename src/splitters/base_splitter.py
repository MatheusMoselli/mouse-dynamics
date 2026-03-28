"""
Generic splitter for splitting the data into test/training groups.
"""
from abc import abstractmethod
from pathlib import Path
import pandas as pd
from src.dto import ExtractionData, UserDataDto
from src.utils.log_file import log_dataframe_sessions


class BaseSplitter:
    """
    Split the dataset into train/test sets.
    The features should already be extracted at this point
    """

    def __init__(self, is_debug: bool = False):
        """
        Class initialization.
        :param is_debug: Is the splitter being run in debug mode.
        """
        self.is_debug = is_debug

    @abstractmethod
    def split(self, extraction_data: ExtractionData) -> ExtractionData:
        """
        Split the dataset into train/test sets.
        :param extraction_data:  the list of datasets to be split
        :return: The list of train/test sets
        """
        pass

    @staticmethod
    def _write_debug_file(user: UserDataDto) -> None:
        directory_path = Path(f"../datasets/split/user{user.id}")
        directory_path.mkdir(parents=True, exist_ok=True)

        log_dataframe_sessions(directory_path / "training", user.training_sessions)
        log_dataframe_sessions(directory_path / "testing", user.testing_sessions)
"""
Generic splitter for splitting the data into test/training groups.
"""
from typing import Dict
from pandas import DataFrame

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

    def split(self, dataframes_by_users: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Split the dataset into train/test sets.
        :param dataframes_by_users:  the list of datasets to be split
        :return: The list of train/test sets
        """
        pass

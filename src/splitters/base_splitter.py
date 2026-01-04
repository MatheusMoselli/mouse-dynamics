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

    def split(self, dataframes_by_users: Dict[str, DataFrame], is_debug=False) -> Dict[str, DataFrame]:
        """
        Split the dataset into train/test sets.
        :param dataframes_by_users:  the list of datasets to be split
        :param is_debug: If true, will save a parquet file for each user, with its features extracted
        :return: The list of train/test sets
        """
        pass

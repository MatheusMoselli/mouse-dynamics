"""
Class responsible for storing the user`s training and testing data.
Will be used throughout the whole process
"""
import pandas as pd

class UserDataDto:
    _training_df: pd.DataFrame
    _testing_df: pd.DataFrame
    _id: str

    def __init__(self, user_id: str):
        self._id = user_id

    @property
    def training_dataframe(self) -> pd.DataFrame:
        return self._training_df

    @property
    def testing_dataframe(self) -> pd.DataFrame:
        return self._testing_df

    @property
    def id(self) -> str:
        return self._id

    @training_dataframe.setter
    def training_dataframe(self, training_dataframe: pd.DataFrame):
        self._training_df = training_dataframe

    @testing_dataframe.setter
    def testing_dataframe(self, test_dataframe: pd.DataFrame):
        self._testing_df = test_dataframe

"""
Class responsible for storing the user`s training and testing data.
Will be used throughout the whole process
"""
import pandas as pd
from src.dto.enums import EnumTypeOfSession

class UserDataDto:
    _training_df: pd.DataFrame | None
    _testing_df: pd.DataFrame | None
    _id: str

    def __init__(self, user_id: str):
        self._id = user_id
        self._training_df = None
        self._testing_df = None

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

    def append_dataframe(self, dataframe: pd.DataFrame, session_type: EnumTypeOfSession):
        current_df = self.get_dataframe_by_type(session_type)

        if current_df is None:
            current_df = dataframe
        else:
            current_df = pd.concat([current_df, dataframe], ignore_index=True)

        if session_type == EnumTypeOfSession.TRAINING:
            self._training_df = current_df
        else:
            self._testing_df = current_df

    def get_dataframe_by_type(self, session_type: EnumTypeOfSession) -> pd.DataFrame:
        return self._training_df if session_type == EnumTypeOfSession.TRAINING else self._testing_df

    @testing_dataframe.setter
    def testing_dataframe(self, test_dataframe: pd.DataFrame):
        self._testing_df = test_dataframe

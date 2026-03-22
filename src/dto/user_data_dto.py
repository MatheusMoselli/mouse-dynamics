"""
Class responsible for storing the user`s training and testing data.
Will be used throughout the whole process
"""
import pandas as pd
from src.dto.enums import EnumTypeOfSession

class UserDataDto:
    _training_sessions: dict[str, pd.DataFrame] | None
    _testing_sessions: dict[str, pd.DataFrame] | None
    _id: str

    def __init__(self, user_id: str):
        self._id = user_id
        self._training_sessions = None
        self._testing_sessions = None

    @property
    def training_sessions(self) -> dict[str, pd.DataFrame]:
        return self._training_sessions

    @property
    def testing_sessions(self) -> dict[str, pd.DataFrame]:
        return self._testing_sessions

    @property
    def id(self) -> str:
        return self._id

    @training_sessions.setter
    def training_sessions(self, training_dataframe: dict[str, pd.DataFrame]):
        self._training_sessions = training_dataframe

    @testing_sessions.setter
    def testing_sessions(self, test_dataframe: dict[str, pd.DataFrame]):
        self._testing_sessions = test_dataframe

    def append_session(self, session_name: str, dataframe: pd.DataFrame, session_type: EnumTypeOfSession):
        current_session = self.get_session_by_name(session_name, session_type)

        if current_session is None:
            current_session = dataframe
        else:
            current_session = pd.concat([current_session, dataframe], ignore_index=True)

        if session_type == EnumTypeOfSession.TRAINING:
            self._training_sessions[session_name] = current_session
        else:
            self._testing_sessions[session_name] = current_session

    def get_session_by_name(self, session_name: str, type_of_session: EnumTypeOfSession):
        return self._training_sessions.get(session_name, None) \
            if type_of_session == EnumTypeOfSession.TRAINING \
            else self._testing_sessions.get(session_name, None)


    def get_sessions_by_type(self, session_type: EnumTypeOfSession) -> dict[str, pd.DataFrame]:
        return self._training_sessions if session_type == EnumTypeOfSession.TRAINING else self._testing_sessions

    def is_user_valid(self):
        return self.id is not None \
            and self.training_sessions is not None \
            and len(self.training_sessions) > 0 \
            and self.testing_sessions is not None \
            and len(self.testing_sessions) > 0


"""
Class responsible for storing the user's training and testing data.
Used throughout the whole preprocessing and classification pipeline.
"""
import pandas as pd
from src.dto.enums import EnumTypeOfSession

class UserDataDto:
    _id: str
    _training_sessions: dict[str, pd.DataFrame]
    _testing_sessions: dict[str, pd.DataFrame]

    def __init__(self, user_id: str):
        self._id = user_id
        self._training_sessions = {}
        self._testing_sessions = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def training_sessions(self) -> dict[str, pd.DataFrame]:
        return self._training_sessions

    @property
    def testing_sessions(self) -> dict[str, pd.DataFrame]:
        return self._testing_sessions

    @training_sessions.setter
    def training_sessions(self, value: dict[str, pd.DataFrame]) -> None:
        self._training_sessions = value

    @testing_sessions.setter
    def testing_sessions(self, value: dict[str, pd.DataFrame]) -> None:
        self._testing_sessions = value

    def append_session(
        self,
        session_name: str,
        dataframe: pd.DataFrame,
        session_type: EnumTypeOfSession,
    ) -> None:
        """
        Add rows to a named session. If the session already exists, the new
        rows are concatenated; otherwise a new entry is created.

        :param session_name: Unique identifier for this session
        :param dataframe: Rows to add
        :param session_type: TRAINING or TESTING
        """
        target = (
            self._training_sessions
            if session_type == EnumTypeOfSession.TRAINING
            else self._testing_sessions
        )

        existing = target.get(session_name)

        target[session_name] = (
            dataframe
            if existing is None
            else pd.concat([existing, dataframe], ignore_index=True)
        )

    def get_session_by_name(
        self, session_name: str, type_of_session: EnumTypeOfSession
    ) -> pd.DataFrame | None:
        sessions = self.get_sessions_by_type(type_of_session)
        return sessions.get(session_name)

    def get_sessions_by_type(
        self, session_type: EnumTypeOfSession
    ) -> dict[str, pd.DataFrame]:
        return (
            self._training_sessions
            if session_type == EnumTypeOfSession.TRAINING
            else self._testing_sessions
        )

    def is_user_valid(self) -> bool:
        return (
            self._id is not None
            and len(self._training_sessions) > 0
            and len(self._testing_sessions) > 0
        )
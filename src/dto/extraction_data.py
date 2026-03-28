"""
Class responsible for storing the data for all the users.
"""
from src.dto import UserDataDto


class ExtractionData:
    def __init__(self, users: list[UserDataDto] | None = None):
        self.__users: list[UserDataDto] = list(users) if users is not None else []

    def add_user(self, user: UserDataDto) -> UserDataDto:
        self.__users.append(user)
        return user

    @property
    def users(self) -> list[UserDataDto]:
        return self.__users

    def get_user_by_id(self, user_id: str) -> UserDataDto | None:
        for user in self.__users:
            if user.id == user_id:
                return user
        return None
"""
Class responsible for storing the data for all the users.
"""
from src.dto import UserDataDto


class ExtractionData:
    def __init__(self, users: list[UserDataDto] | None = None):
        """
        :param users: initial user list
        """
        self.__users: list[UserDataDto] = list(users) if users is not None else []

    def add_user(self, user: UserDataDto) -> UserDataDto:
        """
        Add a user to the list
        :param user: user to be added to the list
        :return: user added
        """
        self.__users.append(user)
        return user

    @property
    def users(self) -> list[UserDataDto]:
        """
        :return: user list
        """
        return self.__users

    def get_user_by_id(self, user_id: str) -> UserDataDto | None:
        """
        Get user by id
        :param user_id: id of the user
        :return: user if found, None otherwise
        """
        for user in self.__users:
            if user.id == user_id:
                return user
        return None
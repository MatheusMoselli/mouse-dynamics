"""
Base classifier for better abstraction and dependency injection
"""
from typing import Dict
from sklearn.ensemble import RandomForestClassifier as SkLearnRandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import pandas as pd


class BaseClassifier(ABC):
    """
    Abstraction for all classifiers.
    """

    @abstractmethod
    def fit (self, dataframes_by_users: Dict[str, pd.DataFrame]):
        """
        Fit the user`s datas into the desired classifiers, printing the results in the console.

        :param dataframes_by_users: The user`s dataframes.
        """
        pass

class RandomForestClassifier(BaseClassifier):
    """
    Random Forest Classifier.
    """

    def __init__(self):
        # Create the model
        self.model = SkLearnRandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-2
        )

    def fit(self, dataframes_by_users: Dict[str, pd.DataFrame]):
        """
        Fit the user`s datas into the desired classifiers, printing the results in the console.

        :param dataframes_by_users: The user`s dataframes.
        """

        for user_id, df in dataframes_by_users.items():
            x = df.copy().drop(columns=["authentic"]).dropna()
            y = df.copy().dropna()["authentic"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)

            self.model.fit(x_train, y_train)

            # Test model
            y_prediction = self.model.predict(x_test)
            print("Classification report for classifier " + user_id + ":")
            print(classification_report(y_test, y_prediction))
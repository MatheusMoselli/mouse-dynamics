"""
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various subsamples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    Trees in the forest use the best split strategy.
    see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as SkLearnRandomForestClassifier
from sklearn.metrics import classification_report
from src.classifiers import BaseClassifier
from src.dto import ExtractionData, UserDataDto
import logging

logger = logging.getLogger(__name__)

class RandomForestClassifier(BaseClassifier):
    """
    Custom Random Forest Classifier following the project pattern
    """

    def __init__(self, is_debug: bool = False):
        super().__init__(is_debug)

    def fit(self, extraction_data: ExtractionData):
        """
        Fit the user`s datas into the desired classifiers, printing the results in the console.

        :param extraction_data: The user`s dataframes.
        """
        for user in extraction_data.users:
            data = self._prepare_user_data(user)

            if data is None:
                logger.info(f"User {user.id} skipped (invalid or empty data).")
                continue

            x_train, y_train, x_test, y_test = data

            model = SkLearnRandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-2
            )

            model.fit(x_train, y_train)
            y_prediction = model.predict(x_test)

            print("Classification report for classifier " + user.id + ":")
            print(classification_report(y_test, y_prediction))
"""
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various subsamples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    Trees in the forest use the best split strategy.
    see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""
from sklearn.ensemble import RandomForestClassifier as SkLearnRandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from src.classifiers import BaseClassifier
from src.dto import ExtractionData
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
        Fit the user`s datas into the random forest classifier.

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

            score = model.score(x_test, y_test)
            balanced_score = balanced_accuracy_score(y_test, y_prediction)

            print("Classification report for classifier " + user.id + ":")
            print("Score: " + str(score))
            print("Balanced Score: " + str(balanced_score))
            print(classification_report(y_test, y_prediction))
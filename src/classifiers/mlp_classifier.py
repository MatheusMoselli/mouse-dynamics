"""
Multi-layer Perceptron (MLP) classifier.
This model optimizes the log-loss function using LBFGS or stochastic gradient descent.
see: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""
from sklearn.neural_network import MLPClassifier as SkLearnMLPClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from src.classifiers import BaseClassifier
from src.dto import ExtractionData
import logging

logger = logging.getLogger(__name__)

class MLPClassifier(BaseClassifier):
    """
    Custom MLP classifier following the project pattern
    """

    def __init__(self, is_debug: bool = False):
        super().__init__(is_debug)

    def fit(self, extraction_data: ExtractionData):
        """
        Fit the user`s datas into the MLP classifier.

        :param extraction_data: The user`s dataframes.
        """
        for user in extraction_data.users:
            data = self._prepare_user_data(user)

            if data is None:
                logger.info(f"User {user.id} skipped (invalid or empty data).")
                continue

            x_train, y_train, x_test, y_test = data

            model = SkLearnMLPClassifier(random_state=42)

            model.fit(x_train, y_train)
            y_prediction = model.predict(x_test)

            score = model.score(x_test, y_test)
            balanced_score = balanced_accuracy_score(y_test, y_prediction)

            print("Classification report for classifier " + user.id + ":")
            print("Score: " + str(score))
            print("Balanced Score: " + str(balanced_score))
            print(classification_report(y_test, y_prediction))
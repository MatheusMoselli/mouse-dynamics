"""
Classifier implementing the k-nearest neighbors vote.
see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from src.classifiers import BaseClassifier
from src.dto import ExtractionData
import logging

logger = logging.getLogger(__name__)

class KNNClassifier(BaseClassifier):
    """
    Custom KNN classifier following the project pattern
    """

    def __init__(self, is_debug: bool = False):
        super().__init__(is_debug)

    def fit(self, extraction_data: ExtractionData):
        """
        Fit the user`s datas into the KNN classifier.

        :param extraction_data: The user`s dataframes.
        """
        for user in extraction_data.users:
            data = self._prepare_user_data(user)

            if data is None:
                logger.info(f"User {user.id} skipped (invalid or empty data).")
                self._experiment_logger.increase_skipped_users_amount_log()
                continue

            x_train, y_train, x_test, y_test = data

            if self.is_debug:
                model = KNeighborsClassifier(
                    n_neighbors=5,
                    weights='distance',
                    algorithm='auto',
                    n_jobs=1
                )
            else:
                study_name = f"knn_user_{user.id}"

                model = self._get_best_model(
                    x_train,
                    y_train,
                    study_name
                )

            model.fit(x_train, y_train)
            y_prediction = model.predict(x_test)

            score = model.score(x_test, y_test)
            balanced_score = balanced_accuracy_score(y_test, y_prediction)

            if self.is_debug:
                print("Classification report for classifier " + user.id + ":")
                print("Score: " + str(score))
                print("Balanced Score: " + str(balanced_score))
                print(classification_report(y_test, y_prediction))

            self._log_user_result(
                user_id=user.id,
                y_test=y_test,
                y_pred=y_prediction,
                score=score,
                balanced_score=balanced_score,
                best_params=model.get_params(),
            )

    def _objective(self,
                   trial: optuna.Trial,
                   x_train: np.ndarray,
                   y_train: np.ndarray) -> float:
        """
        Defines the function to run in the trials
        :param trial: current trial
        :param x_train: training data
        :param y_train: training labels
        :return: the mean of scores in the trial
        """
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),

            "weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),

            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "brute"]
            ),
        }

        metric = trial.suggest_categorical(
            "metric",
            ["euclidean", "manhattan", "chebyshev", "minkowski"]
        )

        params["metric"] = metric

        if metric == "minkowski":
            params["p"] = trial.suggest_int("p", 1, 4)

        model = KNeighborsClassifier(
            **params,
            n_jobs=1
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            error_score=0.0,
            n_jobs=1
        )

        return float(scores.mean())

    def _train_best_model(
            self,
            best_params: dict,
            x_train: pd.DataFrame,
            y_train: pd.Series,
    ):
        """
        Train the final model.
        :param best_params: the params to use in the model
        :param x_train: training data
        :param y_train: training labels
        :return: the best model
        """
        return KNeighborsClassifier(
            n_neighbors=best_params["n_neighbors"],
            weights=best_params["weights"],
            metric=best_params["metric"],
            p=best_params.get("p", 2),
            algorithm=best_params["algorithm"],
            leaf_size=best_params["leaf_size"]
        )
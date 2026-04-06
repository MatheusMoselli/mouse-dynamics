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

NUMBER_OF_TRIALS = 3
CROSS_VALIDATION_FOLDS = 3

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
                continue

            x_train, y_train, x_test, y_test = data

            model = self._get_best_model(x_train, y_train)

            model.fit(x_train, y_train)
            y_prediction = model.predict(x_test)

            score = model.score(x_test, y_test)
            balanced_score = balanced_accuracy_score(y_test, y_prediction)

            print("Classification report for classifier " + user.id + ":")
            print("Score: " + str(score))
            print("Balanced Score: " + str(balanced_score))
            print(classification_report(y_test, y_prediction))

    def _objective(self,
                   trial: optuna.Trial,
                   x_train: np.ndarray,
                   y_train: np.ndarray) -> float:
        """
        Espaço de busca para KNN.

        Notas:
        - 'algorithm' depende de 'metric': ball_tree/kd_tree não suportam minkowski com 'p' fracionário
        - weights='distance' tende a ajudar em dados biométricos com classes desbalanceadas
        """
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "chebyshev", "minkowski"]),
            "p": trial.suggest_int("p", 1, 4),
            "algorithm": trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree", "brute"]),
            "leaf_size": trial.suggest_int("leaf_size", 10, 60),
        }

        model = KNeighborsClassifier(**params, n_jobs=-1)

        cv = StratifiedKFold(n_splits=NUMBER_OF_TRIALS, shuffle=True, random_state=42)

        scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            error_score=0.0,
        )

        return float(scores.mean())

    def _train_best_model(
            self,
            best_params: dict,
            x_train: pd.DataFrame,
            y_train: pd.Series,
    ):
        return KNeighborsClassifier(
            n_neighbors=best_params["n_neighbors"],
            weights=best_params["weights"],
            metric=best_params["metric"],
            p=best_params.get("p", 2),
            algorithm=best_params["algorithm"],
            leaf_size=best_params["leaf_size"],
            n_jobs=-1,
        )
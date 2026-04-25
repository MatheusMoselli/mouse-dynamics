"""
Multi-layer Perceptron (MLP) classifier.
This model optimizes the log-loss function using LBFGS or stochastic gradient descent.
see: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
                self._experiment_logger.increase_skipped_users_amount_log()
                continue

            x_train, y_train, x_test, y_test = data

            if self.is_debug:
                model = SkLearnMLPClassifier(
                    hidden_layer_sizes=(50,),   # pequeno
                    activation='relu',
                    solver='adam',
                    alpha=1e-4,
                    batch_size=64,              # menor = mais rápido por época
                    learning_rate_init=1e-3,
                    max_iter=150,               # reduza forte
                    early_stopping=True,        # ESSENCIAL
                    n_iter_no_change=10,
                    random_state=42
                )
            else:
                model = self._get_best_model(x_train, y_train)


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
                   x_train: pd.DataFrame,
                   y_train: pd.Series) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 4)

        params = {
                "hidden_layer_sizes": tuple(
                    trial.suggest_int(f"n_units_l{i}", 32, 512, log=True)
                    for i in range(n_layers)
                ),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
                "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, "auto"]),
                "momentum": trial.suggest_float("momentum", 0.7, 0.99),
        }

        model = SkLearnMLPClassifier(
            **params,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )

        cv = StratifiedKFold(n_splits=self.NUMBER_OF_TRIALS, shuffle=True, random_state=42)

        scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            error_score=0.0,
            n_jobs=-2
        )

        return float(scores.mean())

    def _train_best_model(
            self,
            best_params: dict,
            x_train: pd.DataFrame,
            y_train: pd.Series,
    ) -> SkLearnMLPClassifier:
        """Train the final model."""
        return SkLearnMLPClassifier(
            hidden_layer_sizes=tuple(best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])),
            activation=best_params["activation"],
            solver=best_params["solver"],
            alpha=best_params["alpha"],
            learning_rate=best_params["learning_rate"],
            learning_rate_init=best_params["learning_rate_init"],
            batch_size=best_params["batch_size"],
            momentum=best_params.get("momentum", 0.9),
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )

"""
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various subsamples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    Trees in the forest use the best split strategy.
    see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SkLearnRandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

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
                self._experiment_logger.increase_skipped_users_amount_log()
                continue

            x_train, y_train, x_test, y_test = data

            if self.is_debug:
                model = SkLearnRandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    n_jobs=1,
                    random_state=42
                )
            else:
                study_name = f"user{user.id}"

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
                   x_train: pd.DataFrame,
                   y_train: pd.Series) -> float:
        """
        Defines the function to run in the trials
        :param trial: current trial
        :param x_train: training data
        :param y_train: training labels
        :return: the mean of scores in the trial
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),

            "max_depth": trial.suggest_int("max_depth", 5, 20),

            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),

            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),

            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5]
            ),

            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy"]
            ),

            "bootstrap": trial.suggest_categorical(
                "bootstrap", [True, False]
            ),

            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", None]
            ),
        }
        
        if params["bootstrap"]:
            params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)

        model = SkLearnRandomForestClassifier(
            **params,
            n_jobs=1,
            random_state=42
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        scores = cross_val_score(
            model, x_train, y_train,
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
    ) -> SkLearnRandomForestClassifier:
        """
        Train the final model.
        :param best_params: the params to use in the model
        :param x_train: training data
        :param y_train: training labels
        :return: the best model
        """
        # Remove chaves internas do Optuna que não vão para o sklearn
        params = {k: v for k, v in best_params.items() if k != "class_weight"}
        params["class_weight"] = best_params.get("class_weight")
        params["random_state"] = 42

        model = SkLearnRandomForestClassifier(**params)
        model.fit(x_train, y_train)
        return model

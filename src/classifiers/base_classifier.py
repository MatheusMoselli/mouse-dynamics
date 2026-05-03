"""
Base classifier for better abstraction and dependency injection
"""
import optuna
import pandas as pd
from pandas import Series
from typing import Optional
from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.experiment_logger import ExperimentLogger
from src.dto import ExtractionData, UserDataDto, EnumTypeOfSession

_DROP_COLS = ["authentic"]

class BaseClassifier(ABC):
    """
    Abstraction for all classifiers.
    """

    NUMBER_OF_TRIALS = 15
    _experiment_logger: ExperimentLogger = None

    def __init__(self, is_debug: bool = False):
        """
        Class initialization.
        :param is_debug: Is the classifier being run in debug mode.
        """
        self.is_debug = is_debug

    def set_experiment_logger(self, experiment_logger: ExperimentLogger):
        """
        Set the experiment logger for the classifier (as read-only).
        :param experiment_logger: Experiment logger.
        """
        self._experiment_logger = experiment_logger

    def _log_user_result(
        self,
        user_id: str,
        y_test: Series,
        y_pred: Series,
        score: float,
        balanced_score: float,
        best_params: dict | None = None,
    ) -> None:
        """
        Add a user result to the experiment logger.
        """
        if self._experiment_logger is not None:
            self._experiment_logger.log_user_result(
                user_id=user_id,
                y_test=y_test,
                y_pred=y_pred,
                score=score,
                balanced_score=balanced_score,
                best_params=best_params,
            )

    def _log_skipped_user(self) -> None:
        """Increase the amount of users skipped."""
        if self._experiment_logger is not None:
            self._experiment_logger.increase_skipped_users_amount_log()


    @abstractmethod
    def fit(self, extraction_data: ExtractionData):
        """
        Fit the user`s datas into the desired classifiers, printing the results in the console.

        :param extraction_data: The user`s dataframes.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def _get_best_model(self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            study_name: str):
        """
        Creates the optuna studies and get the best model

        :param x_train: training data
        :param y_train: training labels
        :param study_name: name of the study
        :return: the best model
        """
        x_sample, _, y_sample, _ = train_test_split(
            x_train, y_train, train_size=0.3, stratify=y_train
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            study_name=f"{study_name}.db",
            storage=f"sqlite:///optuna/{study_name}.db",
            load_if_exists=True
        )

        study.optimize(
            lambda trial: self._objective(trial, x_sample, y_sample),
            n_trials=self.NUMBER_OF_TRIALS,
            show_progress_bar=True,
            n_jobs=3
        )

        best_params = study.best_params

        best_model = self._train_best_model(best_params, x_sample, y_sample)
        return best_model

    def _prepare_user_data(
            self,
            user: UserDataDto
    ) -> Optional[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Validate, merge, and split one user's sessions into model-ready arrays.

        :param user: UserDataDto after preprocessing and splitting
        :return: (x_train, y_train, x_test, y_test) or None
        """
        if not user.is_user_valid():
            return None

        train_df = user.merged_sessions(EnumTypeOfSession.TRAINING).dropna()
        test_df = user.merged_sessions(EnumTypeOfSession.TESTING).dropna()

        if train_df.empty or test_df.empty:
            return None

        x_train = train_df.drop(columns=_DROP_COLS)
        y_train = train_df["authentic"]

        x_test = test_df.drop(columns=_DROP_COLS)
        y_test = test_df["authentic"]

        normalized_x_train, normalized_x_test = self._normalize_data(x_train, x_test)

        return normalized_x_train, y_train, normalized_x_test, y_test

    @staticmethod
    def _normalize_data(x_train: pd.DataFrame, x_test: pd.DataFrame):
        """
        Normalize the data before training.
        :param x_train: training data
        :param x_test: testing data
        :return: the normalized data
        """
        scaler = StandardScaler()
        scaler.fit(x_train)

        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        return x_train_scaled, x_test_scaled
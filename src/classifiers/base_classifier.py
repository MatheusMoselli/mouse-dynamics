"""
Base classifier for better abstraction and dependency injection
"""
import optuna
import pandas as pd
from pandas import Series
from typing import Optional
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from src.utils.experiment_logger import ExperimentLogger
from src.dto import ExtractionData, UserDataDto, EnumTypeOfSession

_DROP_COLS = ["authentic"]

class BaseClassifier(ABC):
    """
    Abstraction for all classifiers.
    """

    NUMBER_OF_TRIALS = 30
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
        """"""
        pass

    @abstractmethod
    def _train_best_model(
            self,
            best_params: dict,
            x_train: pd.DataFrame,
            y_train: pd.Series,
    ):
        """"""
        pass

    def _get_best_model(self,
            x_train: pd.DataFrame,
            y_train: pd.Series):
        """"""
        # Cria estudo Optuna — TPE sampler + MedianPruner por padrão
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )

        study.optimize(
            lambda trial: self._objective(trial, x_train, y_train),
            n_trials=self.NUMBER_OF_TRIALS,
            show_progress_bar=True
        )

        best_params = study.best_params

        # Treina modelo final com melhores hiperparâmetros
        best_model = self._train_best_model(best_params, x_train, y_train)
        return best_model

    def _prepare_user_data(
            self,
            user: UserDataDto
    ) -> Optional[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Validate, merge, and split one user's sessions into model-ready arrays.

        Returns None (and logs the reason) when the user should be skipped.
        Otherwise, returns (x_train, y_train, x_test, y_test).

        Concrete classifiers call this at the top of their per-user loop so
        that validation and feature/label splitting are not duplicated across
        every classifier implementation.

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
        scaler = StandardScaler()
        scaler.fit(x_train)

        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        return x_train_scaled, x_test_scaled
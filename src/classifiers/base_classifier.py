"""
Base classifier for better abstraction and dependency injection
"""
import optuna
from sklearn.preprocessing import StandardScaler

from src.dto import ExtractionData, UserDataDto, EnumTypeOfSession
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

_DROP_COLS = ["authentic"]

class BaseClassifier(ABC):
    NUMBER_OF_TRIALS = 30

    """
    Abstraction for all classifiers.
    """
    def __init__(self, is_debug: bool = False):
        """
        Class initialization.
        :param is_debug: Is the classifier being run in debug mode.
        """
        self.is_debug = is_debug

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
            show_progress_bar=True,
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
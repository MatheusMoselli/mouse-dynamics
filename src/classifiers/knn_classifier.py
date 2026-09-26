"""
Classifier implementing the k-nearest neighbors vote.
see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from src.classifiers import BaseClassifier
from src.dto import ExtractionData
import logging

logger = logging.getLogger(__name__)

class KNNClassifier(BaseClassifier):
    """
    Custom KNN classifier following the project pattern
    """

    def __init__(self, is_debug: bool = False, seed_number: int = 1):
        super().__init__(is_debug, seed_number)

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
            # 3-50 permitia k grandes que so encarecem a busca por vizinhos
            # sem ganho consistente de f1_macro nesse tipo de dado;
            # 3-30 cobre a faixa que realmente costuma vencer.
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),

            "weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
        }

        # "brute" foi removido do espaco de busca: e sempre O(n*d) por
        # consulta e, na pratica, "auto" ja cai em brute sozinho quando o
        # dataset/metrica nao favorece KD-Tree -- ou seja, manter "brute"
        # como opcao so forcava o pior caso com mais frequencia sem
        # necessidade. Deixar o sklearn decidir com "auto".
        params["algorithm"] = "auto"

        # "chebyshev" tirado: raramente venceu nos resultados e, com
        # "auto", tende a empurrar para brute force (KD-Tree nao acelera
        # bem essa metrica em alta dimensionalidade).
        metric = trial.suggest_categorical(
            "metric",
            ["euclidean", "manhattan", "minkowski"]
        )

        params["metric"] = metric

        if metric == "minkowski":
            params["p"] = trial.suggest_int("p", 1, 4)

        # leaf_size removido: so afeta construcao/busca em KD-Tree/Ball-Tree
        # e o ganho de tunar isso e marginal comparado ao custo de mais uma
        # dimensao no espaco de busca do Optuna.

        model = KNeighborsClassifier(
            **params,
            n_jobs=1
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed_number)

        # Loop manual (em vez de cross_val_score) para poder reportar o
        # score parcial a cada fold. Sem isso o MedianPruner configurado em
        # _get_best_model nunca tem dado intermediario pra comparar e o
        # trial sempre roda os 3 folds inteiros mesmo quando ja esta
        # visivelmente pior que a mediana dos trials anteriores.
        fold_scores = []
        for step, (train_idx, val_idx) in enumerate(cv.split(x_train, y_train)):
            x_fold_train, x_fold_val = x_train[train_idx], x_train[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            try:
                model.fit(x_fold_train, y_fold_train)
                y_fold_pred = model.predict(x_fold_val)
                fold_score = f1_score(y_fold_val, y_fold_pred, average="macro", zero_division=0)
            except ValueError:
                fold_score = 0.0

            fold_scores.append(fold_score)

            trial.report(float(np.mean(fold_scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

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
            algorithm="auto",
            n_jobs=-1
        )
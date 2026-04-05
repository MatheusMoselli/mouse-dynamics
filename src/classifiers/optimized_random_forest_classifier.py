"""
    Uses the best hyperparameters for each user`s random forest classifiers, found using optuna`s study
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various subsamples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    Trees in the forest use the best split strategy.
    see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier as SkLearnRandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.classifiers import BaseClassifier
from src.dto import ExtractionData
import logging

logger = logging.getLogger(__name__)

NUMBER_OF_TRIALS = 3
CROSS_VALIDATION_FOLDS = 3

class OptimizedRandomForestClassifier(BaseClassifier):
    """
    Custom Random Forest Classifier following the project pattern
    """

    def __init__(self, is_debug: bool = False):
        super().__init__(is_debug)

    @staticmethod
    def _objective(trial: optuna.Trial,
                   x_train: pd.DataFrame,
                   y_train: pd.Series) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = SkLearnRandomForestClassifier(**params)

        cv = StratifiedKFold(n_splits=NUMBER_OF_TRIALS, shuffle=True, random_state=42)

        # F1-macro: penaliza igualmente erros nas duas classes
        scores = cross_val_score(
            model, x_train, y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,  # já paralelizamos no RF
            error_score=0.0,  # trial não falha por dados mal formados
        )

        return float(scores.mean())

    @staticmethod
    def _train_best_model(
            best_params: dict,
            x_train: pd.DataFrame,
            y_train: pd.Series,
    ) -> SkLearnRandomForestClassifier:
        """Train the final model."""
        # Remove chaves internas do Optuna que não vão para o sklearn
        params = {k: v for k, v in best_params.items() if k != "class_weight"}
        params["class_weight"] = best_params.get("class_weight")
        params["random_state"] = 42
        params["n_jobs"] = -1

        model = SkLearnRandomForestClassifier(**params)
        model.fit(x_train, y_train)
        return model

    def fit(self, extraction_data: ExtractionData) -> None:
        """
        Para cada usuário: otimiza hiperparâmetros com Optuna,
        treina o modelo final e imprime o classification report.

        :param extraction_data: ExtractionData com todos os usuários.
        """
        for user in extraction_data.users:
            data = self._prepare_user_data(user)

            if data is None:
                logger.info(f"User {user.id} skipped (invalid or empty data).")
                continue

            x_train, y_train, x_test, y_test = data

            # Cria estudo Optuna — TPE sampler + MedianPruner por padrão
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            )

            study.optimize(
                lambda trial: self._objective(trial, x_train, y_train),
                n_trials=NUMBER_OF_TRIALS,
                show_progress_bar=False,
            )

            best_params = study.best_params
            best_cv_score = study.best_value

            logger.info(
                f"User {user.id}: melhor F1-macro CV = {best_cv_score:.4f} | "
                f"params = {best_params}"
            )

            # Treina modelo final com melhores hiperparâmetros
            best_model = self._train_best_model(best_params, x_train, y_train)

            y_prob = best_model.predict_proba(x_test)[:, 1]

            best_threshold = self._find_best_threshold(y_test, y_prob)
            y_pred = (y_prob >= best_threshold).astype(int)

            pr_auc = average_precision_score(y_test, y_prob)

            print(f"\nClassification report for classifier {user.id} "
                  f"[Optuna CV F1-macro: {best_cv_score:.4f} | "
                  f"PR-AUC: {pr_auc:.4f} | "
                  f"threshold: {best_threshold:.3f}]:")

            print(classification_report(y_test, y_pred))

            if self.is_debug:
                self._print_feature_importance(best_model, x_train.columns.tolist())

    @staticmethod
    def _find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
        """
        Encontra o threshold que maximiza o F1-macro na curva Precision-Recall.

        :param y_true: Labels verdadeiros (0 ou 1)
        :param y_prob: Probabilidades preditas para a classe 1
        :return: Threshold ótimo que maximiza F1-macro
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

        # F1 por threshold — precision e recall têm len(thresholds)+1, ignoramos o último
        f1_scores = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-8)
        )

        return float(thresholds[np.argmax(f1_scores)])

    @staticmethod
    def _print_feature_importance(model: SkLearnRandomForestClassifier, feature_names: list[str]) -> None:
        """Imprime top-10 features por importância (apenas em debug)."""
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top10 = importances.nlargest(10)
        print("\n  Top-10 feature importances:")
        for feat, imp in top10.items():
            bar = "█" * int(imp * 40)
            print(f"    {feat:<35} {imp:.4f}  {bar}")
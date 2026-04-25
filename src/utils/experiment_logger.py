"""
Experiment logger for mouse dynamics ML pipeline.

Saves one JSON record per run.
"""
import json
from pathlib import Path
from pandas import Series
from src.dto import UserResult, ExperimentRecord
from sklearn.metrics import f1_score, precision_score, recall_score

_DEFAULT_OUTPUT_DIR = Path("../outputs/experiments")

class ExperimentLogger:
    """
    Logs one experiment.
    """

    def __init__(
        self,
        classifier_name: str,
        dataset_name: str,
        preprocessor_name: str,
        splitter_name: str,
        is_debug: bool = False,
    ) -> None:
        """
        :param classifier_name: EnumClassifiers value
        :param dataset_name: EnumDatasets value
        :param preprocessor_name: EnumPreprocessors value
        :param splitter_name: EnumSplitters value
        :param is_debug: mirrors Orchestrator.is_debug
        """
        self._record = ExperimentRecord(
            classifier=classifier_name,
            dataset=dataset_name,
            preprocessor=preprocessor_name,
            splitter=splitter_name,
            is_debug=is_debug,
        )
        self._output_dir = Path(_DEFAULT_OUTPUT_DIR)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._finish()
        return False

    def log_user_result(
        self,
        user_id: str,
        y_test: Series,
        y_pred: Series,
        score: float,
        balanced_score: float,
        best_params: dict | None = None,
    ) -> None:
        """
        Record one user's classification result.

        :param user_id: user.id from UserDataDto
        :param y_test: ground-truth labels
        :param y_pred: model predictions
        :param score: model.score(x_test, y_test)
        :param balanced_score: balanced_accuracy_score(y_test, y_pred)
        :param best_params: model.get_params() or Optuna best_params dict
        :return:
        """

        result = UserResult(
            user_id=user_id,
            score=score,
            balanced_score=balanced_score,
            f1_macro=f1_score(y_test, y_pred, average="macro", zero_division=0),
            f1_weighted=f1_score(y_test, y_pred, average="weighted", zero_division=0),
            precision_macro=precision_score(y_test, y_pred, average="macro", zero_division=0),
            recall_macro=recall_score(y_test, y_pred, average="macro", zero_division=0),
            best_params=best_params or {},
        )

        self._record.user_results.append(result.to_dict())

    def increase_skipped_users_amount_log(self) -> None:
        """Increment skipped-user counter (optional, for traceability)."""
        self._record.n_users_skipped += 1

    def _finish(self) -> None:
        """Aggregate metrics and persist. Called automatically by __exit__."""
        self._record.aggregate()
        self._persist()

        print(
            f"\n[ExperimentLogger] run={self._record.run_id} "
            f"| {self._record.classifier} × {self._record.dataset} "
            f"| users={self._record.n_users} skipped={self._record.n_users_skipped} "
            f"| mean_balanced_score={self._record.mean_balanced_score:.4f} "
            f"| mean_f1_macro={self._record.mean_f1_macro:.4f}"
        )

    def _persist(self) -> None:
        filename = f"{self._record.run_id}.json"
        json_path = self._output_dir / filename

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self._record.to_dict(), f, indent=2, ensure_ascii=False)


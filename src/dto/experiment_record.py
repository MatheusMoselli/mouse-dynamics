"""
Class responsible for storing the experiment results.
"""
import uuid
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

@dataclass
class ExperimentRecord:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    classifier: str = ""
    dataset: str = ""
    preprocessor: str = ""
    splitter: str = ""
    is_debug: bool = False
    user_results: list[dict] = field(default_factory=list)

    mean_score: float = 0.0
    mean_balanced_score: float = 0.0
    mean_f1_macro: float = 0.0
    mean_f1_weighted: float = 0.0
    mean_precision_macro: float = 0.0
    mean_recall_macro: float = 0.0
    n_users: int = 0
    n_users_skipped: int = 0

    def aggregate(self) -> None:
        """Recompute aggregate fields from user_results."""
        if not self.user_results:
            return
        keys = ["score", "balanced_score", "f1_macro", "f1_weighted",
                "precision_macro", "recall_macro"]
        arrays = {k: [r[k] for r in self.user_results] for k in keys}
        self.mean_score = float(np.mean(arrays["score"]))
        self.mean_balanced_score = float(np.mean(arrays["balanced_score"]))
        self.mean_f1_macro = float(np.mean(arrays["f1_macro"]))
        self.mean_f1_weighted = float(np.mean(arrays["f1_weighted"]))
        self.mean_precision_macro = float(np.mean(arrays["precision_macro"]))
        self.mean_recall_macro = float(np.mean(arrays["recall_macro"]))
        self.n_users = len(self.user_results)

    def to_dict(self) -> dict:
        """Convert class to dictionary"""
        return asdict(self)

    def to_flat_dict(self) -> dict:
        """Flattened version for DataFrame rows: drops the per-user list."""
        d = self.to_dict()
        d.pop("user_results", None)
        return d
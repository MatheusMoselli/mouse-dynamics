"""
Class responsible for storing the experiment results for a single user
"""
from typing import Any
from dataclasses import dataclass, field, asdict

@dataclass
class UserResult:
    user_id: str
    score: float
    balanced_score: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    best_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert class to dictionary"""
        return asdict(self)

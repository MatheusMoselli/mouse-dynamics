"""
Centralizes the logic for fitting the data and getting the expected results.
"""
from enum import Enum
from .base_classifier import BaseClassifier
from .mlp_classifier import MLPClassifier
from .optimized_random_forest_classifier import OptimizedRandomForestClassifier
from .random_forest_classifier import RandomForestClassifier

class EnumClassifiers(Enum):
    RANDOM_FOREST = "random-forest"
    OPTIMIZED_RANDOM_FOREST = "optimal-random-forest"
    MLP = "mlp"

def load_classifier(classifier_name: EnumClassifiers, is_debug: bool) -> BaseClassifier:
    """
    Factory function to classify and fit the dataframes by user.

    :param classifier_name: Name of the classifier.
    :param is_debug: Is the classifier being run in debug mode.
    :return: A classifier implementation.
    """
    classifiers = {
        EnumClassifiers.RANDOM_FOREST: RandomForestClassifier,
        EnumClassifiers.OPTIMIZED_RANDOM_FOREST: OptimizedRandomForestClassifier,
        EnumClassifiers.MLP: MLPClassifier,
    }

    if classifier_name in classifiers:
        classifier = classifiers[classifier_name](is_debug)
        return classifier
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")


__all__ = [
    "BaseClassifier",
    "EnumClassifiers",
    "load_classifier",
]

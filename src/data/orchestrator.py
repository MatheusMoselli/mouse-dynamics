"""
Orchestrator for code centralization.
Keep all the logic hidden from the result analysis.
"""
from src.data import (
    DatasetsNames,
    BasePreprocessor,
    BaseSplitter,
    load_dataset, BaseClassifier
)

import logging

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator for code centralization.
    Uses builder design pattern for easier usage.
    """
    def __init__(self,
                 preprocessor: BasePreprocessor,
                 splitter: BaseSplitter,
                 classifier: BaseClassifier,
                 is_debug = False):
        self.preprocessor = preprocessor
        self.splitter = splitter
        self.classifier = classifier
        self._is_debug = is_debug
        self.dataframes_by_users = {}

    def _load_dataset(self, dataset_name: DatasetsNames):
        """
        Call the load_dataset method for initializing the dataset.
        :param dataset_name: Enum for choosing which dataset to load.
        :return: self
        """
        logger.info(f"Loading dataset.")
        self.dataframes_by_users = load_dataset(dataset_name)
        return self

    def _preprocess(self):
        """
        Call the preprocessor method for extracting the features from the dataset.
        :return: self
        """
        logger.info(f"Preprocessing.")
        self.dataframes_by_users = self.preprocessor.preprocess(self.dataframes_by_users, is_debug=self._is_debug)
        return self

    def _split(self):
        """
        Call the splitter method for splitting the dataset into test/training.
        :return: self
        """
        logger.info(f"Splitting.")
        self.dataframes_by_users = self.splitter.split(self.dataframes_by_users, is_debug=self._is_debug)
        return self

    def _fit(self):
        """
        Call the classifier method for training the classifier and testing it.
        :return: self
        """
        logger.info(f"Fitting.")
        self.classifier.fit(self.dataframes_by_users)
        return self

    def run(self, dataset_name: DatasetsNames):
        self._load_dataset(dataset_name) \
            ._preprocess() \
            ._split() \
            ._fit()
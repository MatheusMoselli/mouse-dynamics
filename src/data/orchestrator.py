"""
Orchestrator for code centralization.
Keep all the logic hidden from the result analysis.
"""
from src.data import (
    DatasetsNames,
    BasePreprocessor,
    BaseSplitter,
    load_dataset
)

class Orchestrator:
    def __init__(self, preprocessor: BasePreprocessor, splitter: BaseSplitter, fitter, is_debug = False):
        self.preprocessor = preprocessor
        self.splitter = splitter
        self.fitter = fitter
        self._is_debug = is_debug

    def _load_dataset(self, dataset_name: DatasetsNames):
        self.dataframes_by_users = load_dataset(dataset_name)
        return self

    def _preprocess(self):
        self.dataframes_by_users = self.preprocessor.preprocess(self.dataframes_by_users, is_debug=self._is_debug)
        return self

    def _split(self):
        self.dataframes_by_users = self.splitter.split(self.dataframes_by_users, is_debug=self._is_debug)
        return self

    def _fit(self):
        self.fitter.fit(self.dataframes_by_users)
        return self

    def run(self, dataset_name: DatasetsNames):
        self._load_dataset(dataset_name) \
            ._preprocess() \
            ._split() \
            ._fit()
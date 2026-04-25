"""
Orchestrator for code centralization.
Keep all the logic hidden from the result analysis.
"""
from src.dataset_loaders import (load_dataset, EnumDatasets)
from src.classifiers import (load_classifier, EnumClassifiers)
from src.dto import ExtractionData
from src.preprocessors import (load_preprocessor, EnumPreprocessors)
from src.splitters import (load_splitter, EnumSplitters)
import logging
import shutil
import sys
import os

from src.utils.experiment_logger import ExperimentLogger

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
)

root = logging.getLogger()
root.handlers.clear()
root.addHandler(handler)
root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator for code centralization.
    Uses builder design pattern for easier usage.
    """
    def __init__(self,
                 dataset: EnumDatasets,
                 preprocessor: EnumPreprocessors,
                 splitter: EnumSplitters,
                 classifier: EnumClassifiers,
                 is_debug = False):
        self.extraction_data: ExtractionData | None = None
        self._dataset_enum = dataset
        self._preprocessor_enum = preprocessor
        self._splitter_enum = splitter
        self._classifier_enum = classifier
        self._is_debug = is_debug

    @staticmethod
    def __rebuild_directory(directory_path: str):
        try:
            shutil.rmtree(directory_path)
            os.makedirs(directory_path)
        except OSError as e:
            logger.critical(f"Could not recreate the {directory_path} folder! e: {e}")

    def _load_dataset(self):
        """
        Call the load_dataset method for initializing the dataset.
        :return: self
        """
        logger.info(f"Loading dataset.")
        dataset_loader = load_dataset(self._dataset_enum, self._is_debug)
        self.extraction_data = dataset_loader.load()
        return self

    def _preprocess(self):
        """
        Call the preprocessor method for extracting the features from the dataset.
        :return: self
        """
        logger.info(f"Preprocessing.")
        preprocessor = load_preprocessor(self._preprocessor_enum, self._is_debug)
        self.extraction_data = preprocessor.preprocess(self.extraction_data)
        return self

    def _split(self):
        """
        Call the splitter method for splitting the dataset into test/training.
        :return: self
        """
        logger.info(f"Splitting.")
        splitter = load_splitter(self._splitter_enum, self._is_debug)
        self.extraction_data = splitter.split(self.extraction_data)
        return self

    def _fit(self, experiment_logger: ExperimentLogger):
        """
        Call the classifier method for training the classifier and testing it.
        :return: self
        """
        logger.info(f"Fitting.")
        classifier = load_classifier(self._classifier_enum, self._is_debug)
        classifier.set_experiment_logger(experiment_logger)
        classifier.fit(self.extraction_data)
        return self

    def _clean_previous_debug_files(self):
        if not self._is_debug:
            logger.info("Skipping cleaning old debug files.")
            return

        # 1. Deleting base files:
        logger.info(f"Cleaning old debug files - Base dataset.")
        self.__rebuild_directory("../datasets/base")

        # 2. Deleting features files:
        logger.info(f"Cleaning old debug files - Features.")
        self.__rebuild_directory("../datasets/features")

        # 3. Deleting split files:
        logger.info(f"Cleaning old debug files - Split.")
        self.__rebuild_directory("../datasets/split")

    def run(self):
        if self._is_debug:
            self._clean_previous_debug_files()

        with ExperimentLogger(  # ← abre o logger aqui
                classifier_name=self._classifier_enum.value,
                dataset_name=self._dataset_enum.value,
                preprocessor_name=self._preprocessor_enum.value,
                splitter_name=self._splitter_enum.value,
                is_debug=self._is_debug
        ) as experiment_logger:
            self._load_dataset() \
                ._preprocess() \
                ._split() \
                ._fit(experiment_logger)
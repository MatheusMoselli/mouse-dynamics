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
        self.dataset_loader = load_dataset(dataset, is_debug)
        self.preprocessor = load_preprocessor(preprocessor, is_debug)
        self.splitter = load_splitter(splitter, is_debug)
        self.classifier = load_classifier(classifier, is_debug)
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
        self.extraction_data = self.dataset_loader.load()
        return self

    def _preprocess(self):
        """
        Call the preprocessor method for extracting the features from the dataset.
        :return: self
        """
        logger.info(f"Preprocessing.")
        self.extraction_data = self.preprocessor.preprocess(self.extraction_data)
        return self

    def _split(self):
        """
        Call the splitter method for splitting the dataset into test/training.
        :return: self
        """
        logger.info(f"Splitting.")
        self.extraction_data = self.splitter.split(self.extraction_data)
        return self

    def _fit(self):
        """
        Call the classifier method for training the classifier and testing it.
        :return: self
        """
        logger.info(f"Fitting.")
        self.classifier.fit(self.extraction_data)
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

        self._load_dataset() \
            ._preprocess() \
            ._split() \
            ._fit()
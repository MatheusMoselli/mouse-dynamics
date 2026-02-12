from src.classifiers import EnumClassifiers
from src.dataset_loaders import EnumDatasets
from src.orchestrator import Orchestrator
from src.preprocessors import EnumPreprocessors
from src.splitters import EnumSplitters
import logging
import sys

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

if __name__ == "__main__":
    orchestrator = Orchestrator(
        dataset=EnumDatasets.MINECRAFT,
        splitter=EnumSplitters.MINECRAFT,
        classifier=EnumClassifiers.RANDOM_FOREST,
        preprocessor=EnumPreprocessors.KHAN,
        is_debug=True
    )

    orchestrator.run()
from src.classifiers import RandomForestClassifier
from src.orchestrator import Orchestrator
from src.preprocessors import MinecraftPreprocessor
from src.splitters import MouseDynamicsSplitter
from src.utils import DatasetsNames
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
        preprocessor=MinecraftPreprocessor(),
        splitter=MouseDynamicsSplitter(),
        classifier=RandomForestClassifier(),
    )

    orchestrator.run(DatasetsNames.MINECRAFT)
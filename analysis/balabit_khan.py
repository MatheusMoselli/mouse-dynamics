from src.classifiers import EnumClassifiers
from src.dataset_loaders import EnumDatasets
from src.orchestrator import Orchestrator
from src.preprocessors import EnumPreprocessors
from src.splitters import EnumSplitters

if __name__ == "__main__":
    orchestrator = Orchestrator(
        dataset=EnumDatasets.BALABIT,
        splitter=EnumSplitters.FIFTY_FIFTY,
        classifier=EnumClassifiers.RANDOM_FOREST,
        preprocessor=EnumPreprocessors.KHAN,
        is_debug=True
    )

    orchestrator.run()
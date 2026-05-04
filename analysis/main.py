from src.classifiers import EnumClassifiers
from src.dataset_loaders import EnumDatasets
from src.orchestrator import Orchestrator
from src.preprocessors import EnumPreprocessors
from src.splitters import EnumSplitters

ALL_DATASETS = [EnumDatasets.BALABIT, EnumDatasets.MINECRAFT, EnumDatasets.BOGAZICI]
ALL_CLASSIFIERS = [EnumClassifiers.RANDOM_FOREST, EnumClassifiers.MLP, EnumClassifiers.KNN]
ALL_WINDOW_SIZES = [10, 50, 100, 150]

ALL_COMBINATIONS = [
    (dataset, classifier, window_size)
    for dataset in ALL_DATASETS
    for classifier in ALL_CLASSIFIERS
    for window_size in ALL_WINDOW_SIZES
]

if __name__ == '__main__':
    for dataset, classifier, window_size in ALL_COMBINATIONS:
        print("=" * 50)
        print(f"Starting analysis for {dataset.value} - {classifier.value} (window_size = {window_size})")
        print("=" * 50)
        
        orchestrator = Orchestrator(
            dataset=dataset,
            splitter=EnumSplitters.HALF,
            classifier=classifier,
            preprocessor=EnumPreprocessors.KHAN,
            preprocessor_window_size=window_size,
            is_debug=False
        )

        orchestrator.run()
        
        print("=" * 50)
        print(f"Ending analysis for {dataset.value} - {classifier.value} (window_size = {window_size})")
        print("=" * 50)

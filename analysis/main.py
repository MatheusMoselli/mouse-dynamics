from src.classifiers import EnumClassifiers
from src.dataset_loaders import EnumDatasets

ALL_DATASETS = [EnumDatasets.MINECRAFT, EnumDatasets.BALABIT, EnumDatasets.BOGAZICI]
ALL_CLASSIFIERS = [EnumClassifiers.RANDOM_FOREST, EnumClassifiers.MLP, EnumClassifiers.KNN]

ALL_COMBINATIONS = [
    (dataset, classifier)
    for dataset in ALL_DATASETS
    for classifier in ALL_CLASSIFIERS
]

if __name__ == '__main__':
    for dataset, classifier in ALL_COMBINATIONS:
        print("")

from src.classifiers    import EnumClassifiers
from src.dataset_loaders import EnumDatasets
from src.preprocessors  import EnumPreprocessors
from src.splitters      import EnumSplitters
from src.orchestrator   import Orchestrator

ALL_CLASSIFIERS  = [EnumClassifiers.RANDOM_FOREST, EnumClassifiers.MLP, EnumClassifiers.KNN]
ALL_WINDOW_SIZES = [10, 50, 100, 150, 200, 250]

total = len(ALL_CLASSIFIERS) * len(ALL_WINDOW_SIZES)
count = 0

for window_size in ALL_WINDOW_SIZES:
    count += len(ALL_CLASSIFIERS)
    print("=" * 80)
    print(f"[{count}/{total}] window_size={window_size}")
    print("=" * 80)

    orchestrator = Orchestrator(
        dataset=EnumDatasets.BOGAZICI,
        splitter=EnumSplitters.HALF,
        classifiers=ALL_CLASSIFIERS,
        preprocessor_window_size=window_size,
        preprocessor=EnumPreprocessors.KHAN,
        is_debug=False,
    )
    orchestrator.run()

    print(f"Concluído: window_size={window_size}\n")

print("=" * 80)
print("TODOS OS EXPERIMENTOS CONCLUÍDOS")
print("=" * 80)
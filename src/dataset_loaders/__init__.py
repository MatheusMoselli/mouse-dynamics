"""
Centralizes the logic for reading datasets and standardizing the resulting dataframes.
"""
from .base_dataset_loader import BaseDatasetLoader
from .balabit_loader import BalabitLoader
from .minecraft_loader import MinecraftLoader
from enum import Enum

class EnumDatasets(Enum):
    BALABIT = "balabit"
    MINECRAFT = "minecraft"

def load_dataset(dataset_name: EnumDatasets, is_debug: bool) -> BaseDatasetLoader:
    """
    Factory function to load datasets by name.

    :param dataset_name: Name of the dataset
    :param is_debug: Is the dataset loader being run in debug mode
    :return: A dataset loader
    """
    loaders = {
        EnumDatasets.BALABIT: BalabitLoader,
        EnumDatasets.MINECRAFT: MinecraftLoader,
    }

    if dataset_name in loaders:
        loader = loaders[dataset_name](is_debug)
        return loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

__all__ = [
    "BaseDatasetLoader",
    "EnumDatasets",
    "load_dataset"
]
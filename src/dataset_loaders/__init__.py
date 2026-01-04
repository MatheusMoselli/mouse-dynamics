"""
Centralizes the logic for reading datasets and standardizing the resulting dataframes.
"""
from ..utils import DatasetsNames
from .base_dataset_loader import BaseDatasetLoader
from .balabit_loader import BalabitLoader
from .minecraft_loader import MinecraftLoader
from typing import Dict
from pandas import DataFrame


def load_dataset(dataset_name: DatasetsNames) -> Dict[str, DataFrame]:
    """
    Factory function to load datasets by name.

    :param dataset_name: Name of the dataset
    :return: Dictionary mapping user_id to DataFrame
    """
    loaders = {
        DatasetsNames.BALABIT: BalabitLoader,
        DatasetsNames.MINECRAFT: MinecraftLoader,
    }

    if dataset_name in loaders:
        loader = loaders[dataset_name]()
        return loader.load()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

__all__ = [
    "BaseDatasetLoader",
    "BalabitLoader",
    "MinecraftLoader",
    "load_dataset"
]
"""
============================================================================
DATA MODULE - src/data/
============================================================================
Complete implementations for data loading, preprocessing, and validation
for mouse dynamics ML projects.
"""

# ============================================================================
# FILE: src/data/__init__.py
# ============================================================================
"""Data loading and preprocessing utilities for mouse dynamics."""

from .loaders import (
    BaseDataLoader,
    BalabitLoader,
    MinecraftLoader,
    load_dataset,
)
# from .preprocessors import (
#     MouseDynamicsPreprocessor,
#     TrajectorySegmenter,
#     OutlierRemover,
# )
# from .augmentation import MouseDynamicsAugmenter
# from .validation import DataValidator

__all__ = [
    "BaseDataLoader",
    "BalabitLoader",
    "MinecraftLoader",
    "load_dataset",
]

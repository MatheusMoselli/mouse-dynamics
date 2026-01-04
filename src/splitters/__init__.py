"""
Centralizes the test/training split for fitting into multiple classifiers
"""
from .base_splitter import BaseSplitter
from .minecraft_splitter import MouseDynamicsSplitter

__all__ = ["BaseSplitter", "MouseDynamicsSplitter"]
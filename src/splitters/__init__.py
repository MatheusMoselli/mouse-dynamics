"""
Centralizes the test/training split for fitting into multiple classifiers
"""
from enum import Enum
from .base_splitter import BaseSplitter
from .half import HalfSplitter

class EnumSplitters(Enum):
    MINECRAFT = "minecraft"
    FIFTY_FIFTY = "fifty_fifty"
    HALF = "half"

def load_splitter(
    splitter_name: EnumSplitters, 
    is_debug: bool,
    window_size: bool,
    seed_number: int) -> BaseSplitter:
    """
    Factory function to split the datasets between train/test sets.

    :param splitter_name: Name of the splitter.
    :param is_debug: Is the splitter being run in debug mode.
    :return: a splitter implementation.
    """
    splitters = {
        EnumSplitters.HALF: HalfSplitter,
    }

    if splitter_name in splitters:
        splitter = splitters[splitter_name](is_debug, window_size, seed_number)
        return splitter
    else:
        raise ValueError(f"Unknown splitter: {splitter_name}")


__all__ = [
    "BaseSplitter",
    "EnumSplitters",
    "load_splitter"
]
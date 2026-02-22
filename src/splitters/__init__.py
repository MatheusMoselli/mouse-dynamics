"""
Centralizes the test/training split for fitting into multiple classifiers
"""
from enum import Enum
from .base_splitter import BaseSplitter
from .fifty_fifty_splitter import FiftyFiftySplitter
from .minecraft_splitter import MinecraftSplitter

class EnumSplitters(Enum):
    MINECRAFT = "minecraft"
    FIFTY_FIFTY = "fifty_fifty"

def load_splitter(splitter_name: EnumSplitters, is_debug: bool) -> BaseSplitter:
    """
    Factory function to split the datasets between train/test sets.

    :param splitter_name: Name of the splitter.
    :param is_debug: Is the splitter being run in debug mode.
    :return: a splitter implementation.
    """
    splitters = {
        EnumSplitters.MINECRAFT: MinecraftSplitter,
        EnumSplitters.FIFTY_FIFTY: FiftyFiftySplitter,
    }

    if splitter_name in splitters:
        splitter = splitters[splitter_name](is_debug)
        return splitter
    else:
        raise ValueError(f"Unknown splitter: {splitter_name}")


__all__ = [
    "BaseSplitter",
    "EnumSplitters",
    "load_splitter"
]
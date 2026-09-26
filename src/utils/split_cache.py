"""
Small disk cache used to hand off one user's train/test split from
HalfSplitter to the classifiers, one user at a time, without ever holding
every user's split in memory simultaneously.

HalfSplitter writes here (write_merged) and frees the user from RAM right
after. BaseClassifier reads here (read_merged) lazily, only when a user's
data isn't already resident in memory -- so the same classifier code keeps
working unchanged for any splitter that keeps everything in RAM too.

This cache is purely transient: Orchestrator runs are one window_size/seed
combo at a time (split -> fit -> next run), so the next run's split()
naturally overwrites these files before they're read again. No need to key
the path by window_size.

Lives in its own "_cache" subfolder, separate from
BaseSplitter._write_debug_file's debug output, so the two never collide on
filenames even when both point at the same base directory.
"""
from pathlib import Path
from typing import Union

import pandas as pd

_CACHE_SEGMENT = "_cache"
_FILENAME = "_merged.parquet"


def split_dir(
    output_dir: Union[str, Path],
    user_id: str,
    session_type: str,
    seed_number: int,
) -> Path:
    """
    Build the on-disk directory for one user's cached split.
    :param session_type: "training" or "testing".
    """
    return (
        Path(output_dir)
        / f"user{user_id}"
        / _CACHE_SEGMENT
        / session_type
        / f"seed{seed_number}"
    )


def write_merged(directory: Path, df: pd.DataFrame) -> None:
    """Persist a user's merged split to disk as parquet."""
    directory.mkdir(parents=True, exist_ok=True)
    df.to_parquet(directory / _FILENAME, index=False)


def read_merged(directory: Path) -> pd.DataFrame:
    """
    Read back a user's cached merged split.
    Raises FileNotFoundError with the exact path if missing -- fail loud
    instead of silently returning empty/wrong data.
    """
    path = directory / _FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Split cache not found at {path}. Was HalfSplitter.split() run with "
            f"the same output_dir/seed_number before fitting?"
        )
    return pd.read_parquet(path)

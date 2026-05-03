"""
Basic logging for debugging.
"""
from pathlib import Path
from pandas import DataFrame

def log_dataframe_file(file: Path, dataframe: DataFrame) -> None:
    """
    Log a dataframe into a file.
    :param file: path of the file
    :param dataframe: dataframe to log
    """
    file.unlink(missing_ok=True)
    dataframe.to_parquet(file, index=False)

def log_dataframe_sessions(base_path: Path, sessions: dict[str, DataFrame]) -> None:
    """
    Log all sessions in a directory.

    :param base_path: base path
    :param sessions: list of sessions to log
    """
    base_path.mkdir(parents=True, exist_ok=True)

    for name, df in sessions.items():
        session_path = base_path / f"{name}.parquet"
        log_dataframe_file(session_path, df)
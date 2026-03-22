from pathlib import Path
from pandas import DataFrame

def log_dataframe_file(file: Path, dataframe: DataFrame) -> None:
    file.unlink(missing_ok=True)
    dataframe.to_parquet(file, index=False)

def log_dataframe_sessions(base_path: Path, sessions: dict[str, DataFrame]) -> None:
    for name, df in sessions.items():
        session_path = base_path / f"{name}.parquet"
        session_path.mkdir(parents=True, exist_ok=True)

        log_dataframe_file(session_path, df)

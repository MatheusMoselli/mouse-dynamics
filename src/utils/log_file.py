from pathlib import Path
from pandas import DataFrame

def log_dataframe_file(file: Path, dataframe: DataFrame) -> None:
    file.unlink(missing_ok=True)
    dataframe.to_parquet(file, index=False)
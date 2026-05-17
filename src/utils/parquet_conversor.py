from pathlib import Path
from pandas import read_parquet
from src.dto.extraction_data import ExtractionData
from src.dto.user_data_dto import UserDataDto

def prepare_extraction_data_from_parquet(window_size: int) -> ExtractionData:
    split_path = Path("../datasets/split")
    extraction_data = ExtractionData()

    for user in sorted(split_path.glob("user*")):
        user_id = user.stem.replace("user", "")
        user_dto = UserDataDto(user_id)

        user_dto.training_sessions = {"_merged": read_parquet(user / str(window_size) / "training" / "_merged.parquet")}
        user_dto.testing_sessions  = {"_merged": read_parquet(user / str(window_size) / "testing"  / "_merged.parquet")}

        extraction_data.add_user(user_dto)

    return extraction_data
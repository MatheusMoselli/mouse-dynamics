from pathlib import Path
from pandas import read_parquet
from src.dto.extraction_data import ExtractionData
from src.dto.user_data_dto import UserDataDto

def prepare_extraction_data_from_parquet(window_size: int) -> ExtractionData:
    split_path = Path("../datasets/split")
    extraction_data = ExtractionData()
    
    for user in split_path.iterdir():
        user_id = user.stem.replace("user","")
        user_dto = UserDataDto(user_id)
        
        training_parquet = read_parquet(user / "training" / f"{window_size}" / "_merged.parquet")
        user_dto.training_sessions = { "_merged": training_parquet }
        
        testing_parquet = read_parquet(user / "testing" / f"{window_size}" / "_merged.parquet")
        user_dto.testing_sessions = { "_merged": testing_parquet }
        
        extraction_data.add_user(user_dto)
        
    return extraction_data
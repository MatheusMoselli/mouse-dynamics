from pathlib import Path
import pandas as pd
from pandas import DataFrame
from src.data import RandomForestClassifier

if __name__ == "__main__":
    feature_extracted_csv_path = "../../datasets/training"
    users_dfs = {}

    for csv_file in Path(feature_extracted_csv_path).glob("*.xlsx"):
        user_df = pd.read_excel(csv_file)
        user_id = csv_file.stem.replace("user", "")

        users_dfs[user_id] = DataFrame(user_df)

    RandomForestClassifier().fit(users_dfs)
from pandas import DataFrame

from src.data import load_dataset, DatasetsNames, MouseDynamicsExtractor
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    # Initializing the actors

    # Loading the dataset standardized
    data_by_users = load_dataset(DatasetsNames.MINECRAFT)
    #TODO: Convert this in a builder, so I can do load_dataset().preprocess().extract_features()

    # Extracting the features
    for user_id, df in data_by_users.items():
        extractor = MouseDynamicsExtractor(df, remove_duplicate=True)
        feature_extracted_df = extractor.extract_features()

        file_path_str = f"../../datasets/features/minecraft/user{user_id}.xlsx"

        file = Path(file_path_str)
        file.unlink(missing_ok=True)

        feature_extracted_df.to_excel(file_path_str, index=False)
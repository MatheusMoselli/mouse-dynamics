from src.data import MinecraftLoader
from src.data.preprocessors import MouseDynamicsPreprocessor
from src.features.extractors import MouseDynamicsExtractor

if __name__ == "__main__":
    # Initializing the actors
    loader = MinecraftLoader()
    preprocessor = MouseDynamicsPreprocessor()
    extractor = MouseDynamicsExtractor()

    # Loading the dataset standardized
    data_by_users = loader.load()

    # Preprocessing the data
    for user_id, df in data_by_users.items():
        preprocessed_df = preprocessor.preprocess(df)
        feature_extracted_df = extractor.extract_features(preprocessed_df)
        feature_extracted_df.to_excel(f"../../datasets/features/minecraft/user{user_id}.xlsx", index=False)
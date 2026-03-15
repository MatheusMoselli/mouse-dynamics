"""
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various subsamples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    Trees in the forest use the best split strategy.
    see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as SkLearnRandomForestClassifier
from sklearn.metrics import classification_report
from src.classifiers import BaseClassifier
from src.dto import ExtractionData, UserDataDto
import logging

logger = logging.getLogger(__name__)

class RandomForestClassifier(BaseClassifier):
    """
    Custom Random Forest Classifier following the project pattern
    """

    def __init__(self, is_debug: bool = False):
        # Create the model
        super().__init__(is_debug)

        self.model = SkLearnRandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-2
        )

    def fit(self, extraction_data: ExtractionData):
        """
        Fit the user`s datas into the desired classifiers, printing the results in the console.

        :param extraction_data: The user`s dataframes.
        """
        for user in extraction_data.users:
            if not user.is_user_valid():
                logger.info(f"User {user.id} is not valid for fitting")
                continue

            training_dataframe_df = user.training_dataframe.copy().dropna()
            x_train = training_dataframe_df.drop(columns=["authentic","session"])
            y_train = training_dataframe_df["authentic"]

            self.model.fit(x_train, y_train)

            # Test model
            testing_dataframe_df = user.testing_dataframe.copy().dropna()
            x_test = testing_dataframe_df.drop(columns=["authentic","session"])
            y_test = testing_dataframe_df["authentic"]

            y_prediction = self.model.predict(x_test)
            print("Classification report for classifier " + user.id + ":")
            print(classification_report(y_test, y_prediction))

if __name__ == "__main__":
    random_forest = RandomForestClassifier(is_debug=True)
    feature_files_location = Path("../../datasets/training")
    test_files_location = Path("../../datasets/features")

    extraction_data = ExtractionData()

    for user_dir in test_files_location.iterdir():
        user_data = UserDataDto(user_dir.stem.replace("user", ""))
        df = pd.read_parquet(user_dir / "testing.parquet")
        logger.info(f"Test dataframe size for User #{user_data.id}: {len(df)}")

        user_data.testing_dataframe = pd.read_parquet(user_dir / "testing.parquet")
        extraction_data.add_user(user_data)

    for training_file in feature_files_location.iterdir():
        if training_file.suffix != ".parquet":
            continue

        user_id = training_file.stem.replace("user", "")

        logger.info("User ID: " + user_id)

        user_data = extraction_data.get_user_by_id(user_id)
        user_data.training_dataframe = pd.read_parquet(training_file)

    random_forest.fit(extraction_data)
"""
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various subsamples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    Trees in the forest use the best split strategy.
    see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SkLearnRandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from src.classifiers import BaseClassifier
from pandas import DataFrame

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

    def fit(self, dataframes_by_users: Dict[str, DataFrame]):
        """
        Fit the user`s datas into the desired classifiers, printing the results in the console.

        :param dataframes_by_users: The user`s dataframes.
        """
        for user_id, df in dataframes_by_users.items():
            x = df.copy().drop(columns=["authentic"]).dropna()
            y = df.copy().dropna()["authentic"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.5)

            self.model.fit(x_train, y_train)

            # Test model
            y_prediction = self.model.predict(x_test)
            print("Classification report for classifier " + user_id + ":")
            print(classification_report(y_test, y_prediction))


if __name__ == "__main__":
    random_forest = RandomForestClassifier(is_debug=True)
    feature_files_location = Path("../../datasets/training")

    dfs_by_users = {}
    for training_file in feature_files_location.iterdir():
        if training_file.suffix != ".parquet":
            continue

        training_user_id = training_file.stem.replace("user", "")
        dfs_by_users[training_user_id] = pd.read_parquet(training_file)

    random_forest.fit(dfs_by_users)
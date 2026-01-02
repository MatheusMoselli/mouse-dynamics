from typing import Dict
from sklearn.ensemble import RandomForestClassifier as SkLearnRandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

class RandomForestClassifier:
    def __init__(self):
        # Create the model
        self.model = SkLearnRandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-2
        )

    def fit(self, dfs_by_users: Dict[str, pd.DataFrame]):
        for user_id, df in dfs_by_users.items():
            x = df.drop(columns=["authentic"]).dropna()
            y = df.dropna()["authentic"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)

            self.model.fit(x_train, y_train)

            # Test model
            y_prediction = self.model.predict(x_test)
            print("Classification report for classifier " + user_id + ":")
            print(classification_report(y_test, y_prediction))
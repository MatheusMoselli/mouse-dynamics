import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

train_data = pd.read_csv("datasets/training/filtered.csv")
X_train = train_data.drop(columns=["class"])
Y_train = train_data["class"]


test_data = pd.read_csv("datasets/testing/filtered.csv")
X_test = test_data.drop(columns=["class"])
Y_test = test_data["class"]

# Create and train model
model = RandomForestClassifier(random_state=42, n_jobs=-2)
model.fit(X_train, Y_train)

# Test models
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))
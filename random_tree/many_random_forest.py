import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

AMOUNT_USER_DATA = 5_000
AMOUNT_MAIN_USER_DATA = 25_000
AMOUNT_OTHER_USERS_DATA = 2_778

def filter_testing():
    training_df = pd.read_csv("datasets/testing/filtered.csv")

    dfs = []

    for i in range(0, 10):
        ith_df = training_df[training_df['class'] == i].head(AMOUNT_USER_DATA)
        dfs.append(ith_df)

    return pd.concat(dfs, ignore_index=True)

def filter_training(main_class = 1):
    training_df = pd.read_csv("datasets/training/data.csv")

    main_df = training_df[training_df['class'] == main_class].head(AMOUNT_MAIN_USER_DATA)
    dfs = [main_df]

    for i in range(0, 10):
        if i == main_class:
            continue

        ith_df = training_df[training_df['class'] == i].head(AMOUNT_OTHER_USERS_DATA)
        dfs.append(ith_df)

    return pd.concat(dfs, ignore_index=True)

def main():
    for user_class in range(0, 10):
        train_data = filter_training(user_class)
        x_train = train_data.drop(columns=["class"])
        y_train = (train_data["class"] == user_class).astype(int)

        test_data = filter_testing()
        x_test = test_data.drop(columns=["class"])
        y_test = (test_data["class"] == user_class).astype(int)

        # Create and train model
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-2
        )

        model.fit(x_train, y_train)

        # Test models
        y_pred = model.predict(x_test)
        print("Classification report for classifier " + str(user_class) + ":")
        print(classification_report(y_test, y_pred))

main()
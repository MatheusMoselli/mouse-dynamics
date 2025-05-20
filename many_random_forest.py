import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

AMOUNT_TESTING_MAIN_USER_DATA = 5_000
AMOUNT_TESTING_OTHER_USERS_DATA = 556
AMOUNT_TRAINING_MAIN_USER_DATA = 25_000
AMOUNT_TRAINING_OTHER_USERS_DATA = 2_778

def filter_testing(main_class, main_user_size, other_user_size):
    # Equivalent to the master10Test_Extracted
    training_df = pd.read_csv("datasets/testing/data.csv")

    main_df = training_df[training_df['class'] == main_class].head(main_user_size)
    dfs = [main_df]

    for i in range(0, 10):
        if i == main_class:
            continue

        ith_df = training_df[training_df['class'] == i].head(other_user_size)
        dfs.append(ith_df)

    return pd.concat(dfs, ignore_index=True)

def filter_training(main_class, main_user_size, other_user_size):
    # Equivalent to the master10Train_Extracted
    training_df = pd.read_csv("datasets/training/data.csv")

    main_df = training_df[training_df['class'] == main_class].head(main_user_size)
    dfs = [main_df]

    for i in range(0, 10):
        if i == main_class:
            continue

        ith_df = training_df[training_df['class'] == i].head(other_user_size)
        dfs.append(ith_df)

    return pd.concat(dfs, ignore_index=True)

# Train using the Training dataset and test using the Testing dataset
def scenario_b():
    for user_class in range(0, 10):
        train_data = filter_training(
            user_class,
            AMOUNT_TRAINING_MAIN_USER_DATA,
            AMOUNT_TRAINING_OTHER_USERS_DATA
        )

        x_train = train_data.drop(columns=["class", "Unnamed: 0"])
        y_train = (train_data["class"] == user_class).astype(int)

        test_data = filter_testing(
            user_class,
            AMOUNT_TESTING_MAIN_USER_DATA,
            AMOUNT_TESTING_OTHER_USERS_DATA
        )

        x_test = test_data.drop(columns=["class", "Unnamed: 0"])
        y_test = (test_data["class"] == user_class).astype(int)

        # Create and train model
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-2
        )

        model.fit(x_train, y_train)

        # Test models
        y_pred = model.predict(x_test)
        print("Classification report for classifier " + str(user_class) + ":")
        print(classification_report(y_test, y_pred))

# Train and test both using the Training dataset
def scenario_a():
    for user_class in range(0, 10):
        train_data = filter_training(
            user_class,
            AMOUNT_TRAINING_MAIN_USER_DATA,
            AMOUNT_TRAINING_OTHER_USERS_DATA
        )

        x = train_data.drop(columns=["class", "Unnamed: 0"])
        y = (train_data["class"] == user_class).astype(int)

        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3)

        # Create and train model
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-2
        )

        model.fit(x_train, y_train)

        # Test models
        y_pred = model.predict(x_test)
        print("Classification report for classifier " + str(user_class) + ":")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # scenario_a()
    scenario_b()
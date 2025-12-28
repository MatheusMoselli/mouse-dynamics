import pandas as pd
from pathlib import Path
from pandas import DataFrame

class MouseDynamicsSplitter:
    """
    Split the dataset into train/test sets.
    The features should already be extracted at this point
    """

    def split(self):
        feature_extracted_csv_path = "../../datasets/features"
        main_users_features = {}
        support_users_features = {}

        for csv_file in Path(feature_extracted_csv_path).glob("*.xlsx"):
            user_df = pd.read_excel(csv_file)
            user_id = csv_file.stem.replace("user", "")

            if int(user_id) < 15:
                main_users_features[user_id] = DataFrame(user_df)
            else:
                support_users_features[user_id] = DataFrame(user_df)

        for main_user_id, main_user_df in main_users_features.items():
            true_user_training_df = main_user_df.copy()
            true_user_training_df["authentic"] = 1

            authentic_df_size = len(true_user_training_df)
            amount_of_support_users = len(support_users_features)
            non_authentic_df_sizes = int(authentic_df_size / amount_of_support_users)

            all_training_dfs = [true_user_training_df]
            for support_user_id, support_user_df in support_users_features.items():
                if main_user_id == support_user_id:
                    continue

                ith_false_user_training_df = support_user_df.head(non_authentic_df_sizes).copy()
                ith_false_user_training_df["authentic"] = 0
                all_training_dfs.append(ith_false_user_training_df)

            file_path_str = f"../../datasets/training/user{main_user_id}.xlsx"

            file = Path(file_path_str)
            file.unlink(missing_ok=True)

            final_training_df = pd.concat(all_training_dfs, ignore_index=True)
            final_training_df.to_excel(file_path_str, index=False)



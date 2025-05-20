import pandas as pd

training_df = pd.read_csv("datasets/training/data.csv")

AMOUNT_MAIN_USER_DATA = 25_000
AMOUNT_OTHER_USERS_DATA = 2_778

main_df = training_df[training_df['class'] == 0].head(AMOUNT_MAIN_USER_DATA)
dfs = [main_df]

for i in range(1, 10):
    ith_df = training_df[training_df['class'] == i].head(AMOUNT_OTHER_USERS_DATA)
    dfs.append(ith_df)

df_combined = pd.concat(dfs, ignore_index=True)
df_combined.to_csv("datasets/training/filtered.csv", index=False)

import pandas as pd

training_df = pd.read_csv("datasets/testing/data.csv")

AMOUNT_USER_DATA = 5_000

dfs = []

for i in range(0, 10):
    ith_df = training_df[training_df['class'] == i].head(AMOUNT_USER_DATA)
    dfs.append(ith_df)

df_combined = pd.concat(dfs, ignore_index=True)
df_combined.to_csv("datasets/testing/filtered.csv", index=False)

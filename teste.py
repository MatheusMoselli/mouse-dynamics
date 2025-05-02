import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load all training files (user 7 data)
training_path = Path("datasets/training")

train_dfs = list()

for f in training_path.iterdir():
    if f.is_file():
        df = pd.read_csv(f)[['x', 'y']]
        train_dfs.append(df)

train_df = pd.concat(train_dfs, ignore_index=True)

centroid = train_df.mean().values  # [mean_x, mean_y]
print("Centroid", centroid)

# Define a distance threshold (can be tuned based on known user data)
# For example: use 95th percentile of training distances as a threshold
train_distances = np.linalg.norm(train_df[['x', 'y']].values - centroid, axis=1)
threshold = np.percentile(train_distances, 95)
print("Threshold", threshold)

# Evaluate test files
test_path = Path("datasets/test")

all_sessions = list()
illegals_sessions = list()
legal_sessions = list()

for f in test_path.iterdir():
    if f.is_file():
        df = pd.read_csv(f)[['x', 'y']]
        df_filtered = df[(df['x'] < 50000) & (df['y'] < 50000)]
        if df_filtered.empty:
            print(f"{f.name}: No valid data.")
            continue

        test_coords = df_filtered[['x', 'y']].values
        distances = np.linalg.norm(test_coords - centroid, axis=1)
        avg_distance = distances.mean()

        # Compare with threshold
        if avg_distance <= threshold:
            legal_sessions.append(f.name)
        else:
            illegals_sessions.append(f.name)

        all_sessions.append(f.name)



results_df = pd.read_csv("datasets/results.csv")

df_filtered_results = results_df[(results_df['filename']).isin(all_sessions)]
print("filtered_results_len: ", len(df_filtered_results))

actual_illegal_sessions = df_filtered_results[(df_filtered_results['is_illegal'] == True)]
print("actual_illegal_sessions: ", len(actual_illegal_sessions))

found_illegal_sessions = actual_illegal_sessions[actual_illegal_sessions['filename'].isin(illegals_sessions)]
print("found_illegal_sessions: ", len(found_illegal_sessions))

actual_legal_sessions = df_filtered_results[(df_filtered_results['is_illegal'] == False)]
false_positives_sessions = actual_legal_sessions[actual_legal_sessions['filename'].isin(illegals_sessions)]
print("false_positives_len: ", len(false_positives_sessions))

print("\n")

print("all illegal sessions found: ", len(illegals_sessions))
print("all legal sessions found: ", len(legal_sessions))
print("all sessions: ", len(all_sessions))
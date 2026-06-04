
import pandas as pd
import numpy as np

print("Loading EdNet dataset sample...")

# =========================
# LOAD SMALL SAMPLE
# =========================

df = pd.read_csv(
    "../../ednetkt1/content/drive/MyDrive/EdNET-KT1/processed_data_03.csv",
    nrows=500000
)

print("\nDataset Loaded Successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

print("\nDataset Preview:")
print(df.head())

# =========================
# SORT TEMPORALLY
# =========================

print("\nSorting interactions by time...")

df = df.sort_values(
    by=["student_id", "timestamp"]
)

# =========================
# FEATURE ENGINEERING
# =========================

print("\nCreating temporal behavioral features...")

# Previous correctness
df["previous_correct"] = (
    df.groupby("student_id")["correct"]
    .shift(1)
)

df["previous_correct"] = (
    df["previous_correct"]
    .fillna(0)
)

# Rolling accuracy
df["rolling_accuracy"] = (
    df.groupby("student_id")["correct"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(0, drop=True)
)

# Response speed
df["fast_response"] = (
    df["time_taken"] < 15
).astype(int)

# Long inactivity
df["long_inactive"] = (
    df["log_delta"] > 60
).astype(int)

# Cognitive difficulty
df["cognitive_load"] = (
    (
        df["time_taken"] > 30
    )
    &
    (
        df["correct"] == 0
    )
).astype(int)

# Confusion behavior
df["confusion_score"] = (
    (
        df["correct"] == 0
    ).astype(int)
    +
    (
        df["time_taken"] > 35
    ).astype(int)
)

# Attention proxy
df["attention_score"] = (
    (
        1
        -
        (
            df["log_delta"] / (
                df["log_delta"].max() + 1
            )
        )
    )
)

# =========================
# STUDENT SEQUENCES
# =========================

print("\nCreating sequential student interactions...")

sequence_features = [
    "problem_id",
    "time_taken",
    "correct",
    "log_delta",
    "rolling_accuracy",
    "cognitive_load",
    "confusion_score",
    "attention_score"
]

student_sequences = []

grouped = df.groupby("student_id")

for student_id, group in grouped:

    group = group.head(50)

    if len(group) < 10:
        continue

    sequence = group[
        sequence_features
    ].values.tolist()

    student_sequences.append({
        "student_id": student_id,
        "sequence_length": len(sequence),
        "sequence": sequence
    })

# =========================
# CREATE FINAL DATAFRAME
# =========================

sequence_df = pd.DataFrame(student_sequences)

print("\nSequence Dataset Preview:")
print(sequence_df.head())

print("\nTotal Student Sequences:")
print(len(sequence_df))

# =========================
# SAVE PREPROCESSED DATA
# =========================

sequence_df.to_pickle(
    "ednet_sequences.pkl"
)

print("\nPreprocessed sequences saved successfully!")

print("\nEdNet preprocessing completed!")


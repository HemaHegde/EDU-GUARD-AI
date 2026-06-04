import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Loading sequence dataset...")

# =========================
# LOAD SEQUENCE DATA
# =========================

with open("ednet_sequences.pkl", "rb") as f:
    sequence_df = pickle.load(f)

print("\nDataset Loaded Successfully!")

print("\nDataset Shape:")
print(sequence_df.shape)

# =========================
# LOAD TRAINED MODELS
# =========================

print("\nLoading trained models...")

lstm_model = load_model("lstm_cognitive_model.h5")

gru_model = load_model("gru_cognitive_model.h5")

cnn_model = load_model("temporal_cnn_model.h5")

print("\nAll models loaded successfully!")

# =========================
# PREPARE SEQUENCES
# =========================

MAX_SEQUENCE_LENGTH = 100

sequences = sequence_df["sequence"].values

X = pad_sequences(
    sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    dtype="float32",
    padding="post",
    truncating="post"
)

print("\nPadded Sequence Shape:")
print(X.shape)

# =========================
# MODEL PREDICTIONS
# =========================

print("\nGenerating predictions...")

lstm_predictions = lstm_model.predict(X)

gru_predictions = gru_model.predict(X)

cnn_predictions = cnn_model.predict(X)

# =========================
# ENSEMBLE PREDICTION
# =========================

ensemble_prediction = (
    lstm_predictions.flatten()
    + gru_predictions.flatten()
    + cnn_predictions.flatten()
) / 3

# =========================
# COGNITIVE SCORES
# =========================

print("\nGenerating cognitive intelligence scores...")

attention_score = (
    1 - ensemble_prediction
) * 100

confusion_score = (
    ensemble_prediction * 100
)

# =========================
# BOREDOM SCORE
# =========================

boredom_score = []

for seq in sequences:

    seq = np.array(seq)

    inactivity_ratio = seq[:, 5].mean()

    boredom_score.append(
        inactivity_ratio * 100
    )

boredom_score = np.array(
    boredom_score
)

# =========================
# COGNITIVE OVERLOAD SCORE
# =========================

cognitive_overload_score = []

for seq in sequences:

    seq = np.array(seq)

    incorrect_ratio = (
        1 - seq[:, 2].mean()
    )

    skip_ratio = seq[:, 6].mean()

    overload = (
        incorrect_ratio
        + skip_ratio
    ) / 2

    cognitive_overload_score.append(
        overload * 100
    )

cognitive_overload_score = np.array(
    cognitive_overload_score
)

# =========================
# FINAL DATAFRAME
# =========================

results_df = pd.DataFrame({

    "student_id":
    sequence_df["student_id"],

    "attention_score":
    attention_score,

    "confusion_score":
    confusion_score,

    "boredom_score":
    boredom_score,

    "cognitive_overload_score":
    cognitive_overload_score,

    "lstm_prediction":
    lstm_predictions.flatten(),

    "gru_prediction":
    gru_predictions.flatten(),

    "cnn_prediction":
    cnn_predictions.flatten(),

    "ensemble_prediction":
    ensemble_prediction
})

# =========================
# RISK CATEGORY
# =========================

def classify_risk(score):

    if score < 30:
        return "Low Risk"

    elif score < 70:
        return "Moderate Risk"

    else:
        return "High Risk"

results_df["risk_category"] = results_df[
    "confusion_score"
].apply(classify_risk)

# =========================
# SAVE RESULTS
# =========================

results_df.to_csv(
    "cognitive_behavior_results.csv",
    index=False
)

print("\nCognitive intelligence results saved!")

# =========================
# PREVIEW RESULTS
# =========================

print("\nResults Preview:")

print(
    results_df.head(10)
)

# =========================
# SUMMARY STATISTICS
# =========================

print("\nAverage Attention Score:")
print(
    results_df["attention_score"].mean()
)

print("\nAverage Confusion Score:")
print(
    results_df["confusion_score"].mean()
)

print("\nAverage Boredom Score:")
print(
    results_df["boredom_score"].mean()
)

print("\nAverage Cognitive Overload Score:")
print(
    results_df[
        "cognitive_overload_score"
    ].mean()
)

print("\nRisk Distribution:")
print(
    results_df[
        "risk_category"
    ].value_counts()
)

print("\nCognitive Intelligence Engine Completed Successfully!")
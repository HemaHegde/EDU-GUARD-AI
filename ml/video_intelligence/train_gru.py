import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU,
    Dense,
    Dropout,
    Masking
)

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping

print("Loading sequence dataset...")

# =========================
# LOAD DATA
# =========================

with open("ednet_sequences.pkl", "rb") as f:
    sequence_df = pickle.load(f)

print("\nDataset Loaded!")

print("\nDataset Shape:")
print(sequence_df.shape)

# =========================
# PREPARE SEQUENCES
# =========================

sequences = sequence_df["sequence"].values

# =========================
# CREATE TARGETS
# =========================

print("\nCreating attention risk labels...")

targets = []

for seq in sequences:

    seq = np.array(seq)

    incorrect_ratio = 1 - seq[:, 2].mean()

    inactivity_ratio = seq[:, 5].mean()

    skip_ratio = seq[:, 6].mean()

    overload_score = (
        incorrect_ratio
        + inactivity_ratio
        + skip_ratio
    ) / 3

    if overload_score >= 0.5:
        targets.append(1)
    else:
        targets.append(0)

targets = np.array(targets)

print("\nTarget Distribution:")
print(pd.Series(targets).value_counts())

# =========================
# PAD SEQUENCES
# =========================

MAX_SEQUENCE_LENGTH = 100

X = pad_sequences(
    sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    dtype="float32",
    padding="post",
    truncating="post"
)

y = targets

print("\nSequence Shape:")
print(X.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# BUILD GRU MODEL
# =========================

print("\nBuilding GRU model...")

model = Sequential()

model.add(
    Masking(
        mask_value=0.0,
        input_shape=(MAX_SEQUENCE_LENGTH, 8)
    )
)

model.add(
    GRU(
        64,
        return_sequences=False
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# =========================
# EARLY STOPPING
# =========================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# =========================
# TRAIN MODEL
# =========================

print("\nTraining GRU model...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=32,
    callbacks=[early_stopping]
)

# =========================
# EVALUATE MODEL
# =========================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nGRU Test Accuracy:")
print(accuracy)

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test)

predicted_classes = (
    predictions > 0.5
).astype(int)

# =========================
# SAVE MODEL
# =========================

print("\nSaving GRU model...")

model.save("gru_cognitive_model.h5")

print("\nGRU model saved successfully!")

# =========================
# SAVE PREDICTIONS
# =========================

prediction_df = pd.DataFrame({
    "actual": y_test,
    "predicted_probability": predictions.flatten(),
    "predicted_class": predicted_classes.flatten()
})

prediction_df.to_csv(
    "gru_predictions.csv",
    index=False
)

print("\nPredictions saved!")

print("\nSample Predictions:")
print(prediction_df.head(10))

print("\nGRU Cognitive Intelligence Completed!")
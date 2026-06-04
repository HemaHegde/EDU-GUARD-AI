import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Masking
)

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping

print("Loading sequence dataset...")

# =========================
# LOAD SEQUENCE DATA
# =========================

with open("ednet_sequences.pkl", "rb") as f:
    sequence_df = pickle.load(f)

print("\nDataset Loaded Successfully!")

print("\nDataset Shape:")
print(sequence_df.shape)

print("\nDataset Preview:")
print(sequence_df.head())

# =========================
# PREPARE SEQUENCES
# =========================

print("\nPreparing sequences...")

sequences = sequence_df["sequence"].values

# =========================
# CREATE TARGET LABELS
# =========================

print("\nCreating cognitive risk labels...")

targets = []

for seq in sequences:

    seq = np.array(seq)

    # confusion proxy
    incorrect_ratio = 1 - seq[:, 2].mean()

    # inactivity proxy
    inactivity_ratio = seq[:, 5].mean()

    # fast skipping proxy
    skip_ratio = seq[:, 6].mean()

    # cognitive overload score
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

print("\nPadding sequences...")

MAX_SEQUENCE_LENGTH = 100

X = pad_sequences(
    sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    dtype="float32",
    padding="post",
    truncating="post"
)

y = targets

print("\nPadded Sequence Shape:")
print(X.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# =========================
# BUILD LSTM MODEL
# =========================

print("\nBuilding LSTM model...")

model = Sequential()

# Ignore padded zeros
model.add(
    Masking(
        mask_value=0.0,
        input_shape=(MAX_SEQUENCE_LENGTH, 8)
    )
)

# LSTM Layer
model.add(
    LSTM(
        64,
        return_sequences=False
    )
)

# Dropout
model.add(
    Dropout(0.3)
)

# Dense Layer
model.add(
    Dense(
        32,
        activation="relu"
    )
)

# Output Layer
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

print("\nTraining LSTM model...")

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

print("\nTest Accuracy:")
print(accuracy)

# =========================
# GENERATE PREDICTIONS
# =========================

print("\nGenerating predictions...")

predictions = model.predict(X_test)

predicted_classes = (
    predictions > 0.5
).astype(int)

# =========================
# SAVE MODEL
# =========================

print("\nSaving trained model...")

model.save("lstm_cognitive_model.h5")

print("\nModel saved successfully!")

# =========================
# SAVE TEST PREDICTIONS
# =========================

prediction_df = pd.DataFrame({
    "actual": y_test,
    "predicted_probability": predictions.flatten(),
    "predicted_class": predicted_classes.flatten()
})

prediction_df.to_csv(
    "lstm_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")

# =========================
# SAMPLE OUTPUTS
# =========================

print("\nSample Predictions:")
print(prediction_df.head(10))

print("\nLSTM Cognitive Intelligence Training Completed!")
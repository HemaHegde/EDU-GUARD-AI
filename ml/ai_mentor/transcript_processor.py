import pandas as pd
import numpy as np
import re

print("Starting AI Mentor Transcript Pipeline...")

# =========================
# SAMPLE EDUCATIONAL CONTENT
# =========================

transcripts = [

    {
        "topic": "Neural Networks",
        "transcript":
        """
        Neural networks are machine learning models inspired by the human brain.
        They contain neurons, layers, activation functions, and weights.
        Backpropagation is used to update weights during training.
        Deep learning uses multiple hidden layers.
        """
    },

    {
        "topic": "Database Management Systems",
        "transcript":
        """
        Database management systems store and organize data efficiently.
        SQL is used for querying relational databases.
        Transactions ensure consistency and reliability.
        """
    },

    {
        "topic": "Operating Systems",
        "transcript":
        """
        Operating systems manage hardware resources and processes.
        Scheduling algorithms optimize CPU performance.
        Memory management handles allocation and paging.
        """
    }

]

# =========================
# CREATE DATAFRAME
# =========================

df = pd.DataFrame(transcripts)

print("\nTranscript Dataset:")
print(df)

# =========================
# CLEAN TEXT
# =========================

print("\nCleaning transcripts...")

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

df["cleaned_transcript"] = (
    df["transcript"]
    .apply(clean_text)
)

print("\nCleaned Transcripts:")
print(
    df[
        [
            "topic",
            "cleaned_transcript"
        ]
    ]
)

# =========================
# CHUNK TRANSCRIPTS
# =========================

print("\nCreating transcript chunks...")

chunk_size = 40

all_chunks = []

for idx, row in df.iterrows():

    words = row[
        "cleaned_transcript"
    ].split()

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        all_chunks.append({

            "topic": row["topic"],

            "chunk": chunk

        })

chunk_df = pd.DataFrame(all_chunks)

print("\nChunk Preview:")
print(chunk_df.head())

print("\nTotal Chunks:")
print(len(chunk_df))

# =========================
# SAVE CHUNKS
# =========================

chunk_df.to_csv(
    "processed_transcript_chunks.csv",
    index=False
)

print(
    "\nTranscript chunks saved successfully!"
)

print(
    "\nAI Mentor Transcript Processing Completed!"
)
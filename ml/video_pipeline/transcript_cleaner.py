import webvtt
import pandas as pd
import os

# =========================
# TRANSCRIPT FILE PATH
# =========================

TRANSCRIPT_PATH = (
    "downloads/"
    "But what is a neural network？ ｜ Deep learning chapter 1.en.vtt"
)

# =========================
# READ VTT FILE
# =========================

captions = webvtt.read(
    TRANSCRIPT_PATH
)

# =========================
# EXTRACT TEXT
# =========================

full_text = ""

for caption in captions:

    full_text += (
        caption.text + " "
    )

# =========================
# CLEAN TEXT
# =========================

full_text = (
    full_text
    .replace("\n", " ")
    .replace("  ", " ")
)

# =========================
# CHUNKING
# =========================

words = full_text.split()

chunk_size = 120

chunks = []

for i in range(
    0,
    len(words),
    chunk_size
):

    chunk = " ".join(

        words[i:i + chunk_size]

    )

    chunks.append(chunk)

# =========================
# SAVE CHUNKS
# =========================

chunk_df = pd.DataFrame({

    "chunk_id":
    range(len(chunks)),

    "chunk":
    chunks

})

output_path = (
    "processed_video_chunks.csv"
)

chunk_df.to_csv(
    output_path,
    index=False
)

# =========================
# FINAL OUTPUT
# =========================

print("\nTranscript cleaned!")

print(
    f"\nTotal Chunks: {len(chunks)}"
)

print(
    "\nSaved File:"
)

print(output_path)
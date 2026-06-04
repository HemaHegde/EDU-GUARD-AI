import pandas as pd
import numpy as np
import pickle
import faiss

from sentence_transformers import (
    SentenceTransformer
)

# =========================
# LOAD CHUNKS
# =========================

chunk_df = pd.read_csv(
    "processed_video_chunks.csv"
)

# =========================
# LOAD EMBEDDING MODEL
# =========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================
# CREATE EMBEDDINGS
# =========================

chunks = chunk_df["chunk"].tolist()

embeddings = model.encode(
    chunks,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

# =========================
# SAVE EMBEDDINGS
# =========================

with open(
    "video_embeddings.pkl",
    "wb"
) as f:

    pickle.dump(
        embeddings,
        f
    )

# =========================
# CREATE FAISS INDEX
# =========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)

# =========================
# SAVE FAISS INDEX
# =========================

faiss.write_index(
    index,
    "video_vector_index.faiss"
)

# =========================
# FINAL OUTPUT
# =========================

print("\nEmbeddings Generated!")

print(
    f"\nTotal Chunks Embedded: {len(chunks)}"
)

print(
    "\nSaved:"
)

print("- video_embeddings.pkl")
print("- video_vector_index.faiss")
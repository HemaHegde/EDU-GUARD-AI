import pandas as pd
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer

import faiss

print("Starting Embedding Engine...")

# =========================
# LOAD TRANSCRIPT CHUNKS
# =========================

chunk_df = pd.read_csv(
    "processed_transcript_chunks.csv"
)

print("\nTranscript Chunks Loaded!")

print("\nChunk Dataset Shape:")
print(chunk_df.shape)

print("\nChunk Preview:")
print(chunk_df.head())

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("\nLoading SentenceTransformer model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("\nEmbedding model loaded successfully!")

# =========================
# CREATE TEXT EMBEDDINGS
# =========================

print("\nGenerating embeddings...")

texts = chunk_df["chunk"].tolist()

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings
).astype("float32")

print("\nEmbedding Shape:")
print(embeddings.shape)

# =========================
# CREATE FAISS INDEX
# =========================

print("\nCreating FAISS vector database...")

embedding_dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    embedding_dimension
)

index.add(embeddings)

print("\nFAISS index created successfully!")

print("\nTotal vectors stored:")
print(index.ntotal)

# =========================
# SAVE FAISS INDEX
# =========================

faiss.write_index(
    index,
    "mentor_vector_index.faiss"
)

print("\nFAISS index saved!")

# =========================
# SAVE EMBEDDINGS
# =========================

with open(
    "transcript_embeddings.pkl",
    "wb"
) as f:

    pickle.dump(
        embeddings,
        f
    )

print("\nEmbeddings saved successfully!")

# =========================
# SAVE METADATA
# =========================

chunk_df.to_csv(
    "mentor_chunk_metadata.csv",
    index=False
)

print("\nChunk metadata saved!")

# =========================
# TEST VECTOR SEARCH
# =========================

print("\nTesting semantic retrieval...")

query = "What are neural networks?"

query_embedding = embedding_model.encode(
    [query]
).astype("float32")

k = 2

distances, indices = index.search(
    query_embedding,
    k
)

print("\nTop Retrieved Chunks:")

for idx in indices[0]:

    print("\n--------------------------------")
    print(chunk_df.iloc[idx]["topic"])
    print("--------------------------------")
    print(chunk_df.iloc[idx]["chunk"])

print("\nEmbedding Engine Completed Successfully!")
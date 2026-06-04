import pandas as pd
import numpy as np
import pickle
import faiss

from sentence_transformers import SentenceTransformer

print("Starting AI Mentor RAG Engine...")

# =========================
# LOAD VECTOR DATABASE
# =========================

print("\nLoading FAISS vector database...")

index = faiss.read_index(
    "mentor_vector_index.faiss"
)

print("\nFAISS database loaded!")

# =========================
# LOAD CHUNK METADATA
# =========================

chunk_df = pd.read_csv(
    "mentor_chunk_metadata.csv"
)

print("\nChunk metadata loaded!")

print("\nTotal Chunks:")
print(len(chunk_df))

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("\nEmbedding model loaded!")

# =========================
# AI MENTOR QUERY FUNCTION
# =========================

def ask_ai_mentor(user_query):

    print("\n=================================")
    print("STUDENT QUESTION:")
    print(user_query)
    print("=================================")

    # ----------------------
    # CREATE QUERY EMBEDDING
    # ----------------------

    query_embedding = embedding_model.encode(
        [user_query]
    ).astype("float32")

    # ----------------------
    # SEARCH VECTOR DATABASE
    # ----------------------

    k = 2

    distances, indices = index.search(
        query_embedding,
        k
    )

    # ----------------------
    # RETRIEVE CONTEXT
    # ----------------------

    retrieved_chunks = []

    print("\nRetrieved Learning Context:")

    for idx in indices[0]:

        topic = chunk_df.iloc[idx]["topic"]

        chunk = chunk_df.iloc[idx]["chunk"]

        retrieved_chunks.append(chunk)

        print("\n--------------------------------")
        print("Topic:", topic)
        print("--------------------------------")
        print(chunk)

    # ----------------------
    # GENERATE RESPONSE
    # ----------------------

    combined_context = " ".join(
        retrieved_chunks
    )

    mentor_response = f"""

AI Mentor Response:

Based on your learning materials,
here is the explanation:

{combined_context}

Guidance:
Focus on understanding the core concepts step by step.
Revise slowly and practice related examples.
You are improving consistently — keep going!

"""

    print("\n=================================")
    print(mentor_response)
    print("=================================")

# =========================
# TEST AI MENTOR
# =========================

ask_ai_mentor(
    "Explain neural networks"
)

ask_ai_mentor(
    "What is SQL?"
)

ask_ai_mentor(
    "How does memory management work?"
)

print("\nAI Mentor RAG Engine Completed Successfully!")
import faiss
import pickle
import numpy as np
import pandas as pd

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
# LOAD FAISS INDEX
# =========================

index = faiss.read_index(
    "video_vector_index.faiss"
)

# =========================
# LOAD EMBEDDING MODEL
# =========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================
# ASK QUESTION
# =========================

while True:

    print("\n===================")
    print("VIDEO AI MENTOR")
    print("===================")

    question = input(
        "\nAsk a question: "
    )

    # -------------------------
    # EXIT
    # -------------------------

    if question.lower() == "exit":

        print("\nExiting AI Mentor...")
        break

    # -------------------------
    # CREATE QUESTION EMBEDDING
    # -------------------------

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding,
        dtype=np.float32
    )

    # -------------------------
    # SEARCH VECTOR DATABASE
    # -------------------------

    distances, indices = index.search(
        question_embedding,
        k=3
    )

    print("\nRelevant Lecture Context:\n")

    combined_context = ""

    for idx in indices[0]:

        chunk = chunk_df.iloc[idx]["chunk"]

        combined_context += chunk + "\n\n"

        print(chunk)
        print("\n-------------------\n")

    # -------------------------
    # AI RESPONSE
    # -------------------------

    print("\nAI Mentor Guidance:\n")

    response = f"""
Based on the lecture:

{combined_context}

Explanation:
This concept is important in deep learning and neural network systems.
Focus on understanding the relationship between neurons, layers,
weights, activation functions, and learning behavior.

Study Tip:
Revise step-by-step and practice small examples regularly.
"""

    print(response)
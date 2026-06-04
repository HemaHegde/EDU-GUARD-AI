
import os
import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================
# FILE PATHS
# =========================

FAISS_PATH = os.path.abspath(

    os.path.join(

        BASE_DIR,

        "../../ml/ai_mentor/mentor_vector_index.faiss"

    )

)

CHUNK_PATH = os.path.abspath(

    os.path.join(

        BASE_DIR,

        "../../ml/ai_mentor/mentor_chunk_metadata.csv"

    )

)

# =========================
# LOAD VECTOR DATABASE
# =========================

index = faiss.read_index(
    FAISS_PATH
)

# =========================
# LOAD CHUNK METADATA
# =========================

chunk_df = pd.read_csv(
    CHUNK_PATH
)

# =========================
# LOAD EMBEDDING MODEL
# =========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================
# ASK MENTOR
# =========================

def ask_mentor(question: str):

    # =========================
    # EMBED QUESTION
    # =========================

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding,
        dtype=np.float32
    )

    # =========================
    # SEARCH FAISS
    # =========================

    distances, indices = index.search(
        question_embedding,
        k=2
    )

    retrieved_chunks = []

    for idx in indices[0]:

        row = chunk_df.iloc[idx]

        retrieved_chunks.append({

            "topic":
            row["topic"],

            "chunk":
            row["chunk"]

        })

    # =========================
    # COMBINE CONTEXT
    # =========================

    combined_context = " ".join(

        [

            item["chunk"]

            for item in retrieved_chunks

        ]

    )

    # =========================
    # HUMAN-LIKE CHATBOT
    # =========================

    lower_question = question.lower()

    # GREETINGS

    if any(word in lower_question for word in [

        "hi",
        "hello",
        "hey"

    ]):

        mentor_response = """

Hi Anika 👋

I'm Aura, your AI learning companion.

I'm here to help you with:

• Studies 📚
• Motivation 🌸
• Stress 💙
• Concepts 🧠
• Productivity ⚡

Ask me anything anytime.
"""

    # STRESS SUPPORT

    elif any(word in lower_question for word in [

        "stress",
        "sad",
        "depressed",
        "tired",
        "anxiety"

    ]):

        mentor_response = """

I'm sorry you're feeling overwhelmed 💙

Remember:

• You do NOT need to solve everything at once.
• Small progress still matters.
• Take a deep breath slowly.

You're stronger than you think 🌸
"""

    # MOTIVATION

    elif any(word in lower_question for word in [

        "motivation",
        "lazy",
        "can't study"

    ]):

        mentor_response = """

It's okay to lose motivation sometimes 🌸

Try this:

1. Study for just 25 minutes
2. Take a 5 minute break
3. Repeat slowly

Starting is always the hardest part ⚡
"""

    # EDUCATIONAL RESPONSE

    else:

        mentor_response = f"""

Here's a simple explanation based on your learning materials 📚

{combined_context[:500]}

✨ Guidance:
Focus on understanding concepts step by step instead of memorizing everything.

You're improving steadily — keep going 🌸
"""

    # =========================
    # RETURN RESPONSE
    # =========================

    return {

        "question":
        question,

        "retrieved_context":
        retrieved_chunks,

        "mentor_response":
        mentor_response.strip()

    }


"""
retrieval.py

Isolates the FAISS vector index + embedding model loading and the
nearest-neighbour lookup used to pull relevant learning material chunks.

Behaviour is unchanged from the original mentor_service.py: same paths,
same model name, same fallback-to-random-embedding behaviour when
sentence_transformers isn't available (so existing index/chunk files
keep working exactly as before).
"""

import os
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception as _st_err:
    print(
        f"WARNING: sentence_transformers could not be loaded "
        f"({_st_err}). AI Mentor will use fallback embeddings."
    )
    SentenceTransformer = None
    _ST_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../ml/ai_mentor/mentor_vector_index.faiss")
)

CHUNK_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../ml/ai_mentor/mentor_chunk_metadata.csv")
)


class LearningMaterialRetriever:
    """
    Thin wrapper around the FAISS index + chunk metadata + embedding
    model, loaded once and reused across calls.
    """

    def __init__(self):
        self._index = faiss.read_index(FAISS_PATH)
        self._chunk_df = pd.read_csv(CHUNK_PATH)
        self._model = None

        if _ST_AVAILABLE:
            try:
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as _model_err:
                print(
                    f"WARNING: Could not load SentenceTransformer model "
                    f"({_model_err}). Using random embeddings."
                )

    def _encode_question(self, question: str) -> np.ndarray:
        if self._model is not None:
            return np.array(self._model.encode([question]), dtype=np.float32)

        # Fallback: random unit vector. FAISS still runs, results are
        # not meaningful — this matches original behaviour exactly,
        # it is not a new fabrication, just preserved as-is.
        vec = np.random.randn(1, 384).astype(np.float32)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        return vec

    def retrieve(self, question: str, k: int = 5) -> List[Dict[str, str]]:
        question_embedding = self._encode_question(question)
        _distances, indices = self._index.search(question_embedding, k)

        retrieved_chunks = []
        for idx in indices[0]:
            row = self._chunk_df.iloc[idx]
            retrieved_chunks.append(
                {"topic": row["topic"], "chunk": row["chunk"]}
            )
        return retrieved_chunks


# Module-level singleton so the index/model are loaded once per process,
# matching the original script's module-level load behaviour.
_retriever = None


def get_retriever() -> LearningMaterialRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LearningMaterialRetriever()
    return _retriever

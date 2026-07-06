"""
retrieval.py

Isolates the FAISS vector index + embedding model loading and the
nearest-neighbour lookup used to pull relevant learning material chunks.

Behaviour is unchanged from the original mentor_service.py: same paths,
same model name, same fallback-to-random-embedding behaviour when
sentence_transformers isn't available (so existing index/chunk files
keep working exactly as before).

SPRINT 8 UPDATE (HALLUCINATION SAFETY HARDENING):
This sprint adds retrieval-level evidence metadata WITHOUT changing the
retrieval algorithm, the index, the embedding model, or which chunks
are selected. The FAISS search call, the ranking it returns, and the
chunk metadata source (mentor_chunk_metadata.csv) are all untouched.
What's new is purely additive per returned chunk:

  1. `source`      - a human-readable citation label for the chunk, so
                     the mentor can tell a student/educator exactly
                     where an answer came from (Req: Evidence
                     Citations). Falls back to the existing `topic`
                     column if the metadata CSV has no dedicated
                     `source` column, since we do not invent a new
                     column or new data that isn't already on disk.
  2. `chunk_id`    - a stable identifier per chunk, used only so
                     confidence.py / mentor_service.py can reason about
                     "how many distinct pieces of evidence support
                     this answer" without re-deriving it from raw
                     dataframe rows. Uses the metadata CSV's own
                     `chunk_id` column if present, else falls back to
                     the FAISS index position (still stable within a
                     given index build).
  3. `similarity_score` - the FAISS neighbour distance, converted to a
                     bounded 0..1 similarity score (see
                     `_similarity_from_distance`). This was previously
                     computed by `_index.search()` and then silently
                     discarded (`_distances`); it is now surfaced so
                     hallucination-safety logic downstream (the
                     SIMILARITY_THRESHOLD gate in mentor_service.py,
                     and the retrieval-quality component of
                     confidence.py) has a real number to reason about,
                     rather than treating "was anything retrieved?" as
                     a yes/no signal.
  4. `raw_distance` - the untransformed FAISS distance, kept alongside
                     `similarity_score` for debugging/audit purposes
                     only; nothing downstream is required to use it.

None of this changes what `LearningMaterialRetriever.retrieve()` was
already doing to select the top-k chunks — it only reports more about
the chunks it was already going to return.

CONFIGURABLE PARAMETERS (Req: Configurable Parameters):
`TOP_K` and `SIMILARITY_THRESHOLD` used to be hardcoded at call sites
(`retriever.retrieve(question, k=5)` in mentor_service.py, and no
threshold existed at all). They are now named constants defined once,
here, since this module owns retrieval concerns; other modules import
them rather than re-declaring their own copies.
"""

import os
from typing import Any, Dict, List

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


# =========================================================
# CONFIGURABLE PARAMETERS (Hallucination Safety Hardening)
# =========================================================
# Single source of truth for retrieval-related tuning knobs. Other
# modules (confidence.py, mentor_service.py) import these rather than
# hardcoding their own copies, so changing retrieval behaviour never
# requires touching more than one file.

# How many nearest-neighbour chunks to pull per question. This was
# previously a hardcoded `k=5` at the mentor_service.py call site.
TOP_K = 5

# Minimum acceptable similarity (0..1, see `_similarity_from_distance`)
# for the TOP retrieved chunk before the mentor is allowed to attempt
# an answer at all. Below this, mentor_service.py returns the fixed
# "Insufficient evidence..." message instead of calling the LLM, per
# the hallucination-safety requirement that we never let a small model
# improvise past weak evidence. This is a starting heuristic value —
# tune it empirically against a labeled set of "should have answered"
# vs. "should have declined" questions for this specific embedding
# model and corpus; it is not derived from any theoretical guarantee.
SIMILARITY_THRESHOLD = 0.35


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

    def _similarity_from_distance(self, distance: float) -> float:
        """
        Converts a raw FAISS neighbour distance into a bounded 0..1
        similarity score, purely for prompt-display and threshold-
        gating purposes. This is a monotonic heuristic transform — it
        does NOT change FAISS's own nearest-neighbour ranking, which
        is untouched.

        FAISS indices can be built with different metrics. We check
        `index.metric_type` and handle both of the two metrics FAISS
        commonly uses:
          - METRIC_INNER_PRODUCT: on (typically L2-normalized)
            sentence embeddings this behaves like cosine similarity in
            roughly [-1, 1]; we rescale to [0, 1].
          - METRIC_L2 (the default assumption if metric_type is
            unavailable/unrecognized, since faiss.read_index does not
            always expose it consistently across FAISS versions):
            smaller distance = more similar, so we map distance 0 -> 1
            and let similarity decay asymptotically toward 0 as
            distance grows, via 1 / (1 + distance).
        This assumption is stated explicitly here (rather than silently
        picked) so anyone tuning SIMILARITY_THRESHOLD later knows
        exactly what the number means for their index type.
        """
        metric_type = getattr(self._index, "metric_type", faiss.METRIC_L2)

        if metric_type == faiss.METRIC_INNER_PRODUCT:
            similarity = (distance + 1.0) / 2.0
        else:
            similarity = 1.0 / (1.0 + max(distance, 0.0))

        return max(0.0, min(1.0, similarity))

    def retrieve(self, question: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Returns up to `k` chunks, ranked exactly as FAISS ranks them
        (unchanged). Each chunk dict now includes, in addition to the
        original `topic` and `chunk` fields:
          - `source`: citation label (see module docstring)
          - `chunk_id`: stable identifier (see module docstring)
          - `similarity_score`: bounded 0..1 similarity (see
            `_similarity_from_distance`)
          - `raw_distance`: the untransformed FAISS distance
        """
        question_embedding = self._encode_question(question)
        distances, indices = self._index.search(question_embedding, k)

        has_source_col = "source" in self._chunk_df.columns
        has_chunk_id_col = "chunk_id" in self._chunk_df.columns

        retrieved_chunks: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices[0]):
            # FAISS pads with -1 when the index holds fewer than k
            # vectors; skip those rather than indexing out of bounds.
            if idx < 0 or idx >= len(self._chunk_df):
                continue

            row = self._chunk_df.iloc[idx]
            raw_distance = float(distances[0][rank])
            similarity_score = self._similarity_from_distance(raw_distance)

            source = row["source"] if has_source_col else row["topic"]
            chunk_id = str(row["chunk_id"]) if has_chunk_id_col else f"chunk_{int(idx)}"

            retrieved_chunks.append({
                "topic": row["topic"],
                "chunk": row["chunk"],
                "source": source,
                "chunk_id": chunk_id,
                "similarity_score": similarity_score,
                "raw_distance": raw_distance,
            })

        return retrieved_chunks


# Module-level singleton so the index/model are loaded once per process,
# matching the original script's module-level load behaviour.
_retriever = None


def get_retriever() -> LearningMaterialRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LearningMaterialRetriever()
    return _retriever

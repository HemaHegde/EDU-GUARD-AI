import os
import faiss
import webvtt
import yt_dlp
import pickle
import numpy as np
import pandas as pd

from sentence_transformers import (
    SentenceTransformer
)

# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VIDEO_PIPELINE_DIR = os.path.join(
    BASE_DIR,
    "../../ml/video_pipeline"
)

DOWNLOAD_DIR = os.path.join(
    VIDEO_PIPELINE_DIR,
    "downloads"
)

# =========================
# LOAD EMBEDDING MODEL
# =========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================
# GLOBAL STORAGE
# =========================

chunk_df = None
index = None

# =========================
# PROCESS YOUTUBE VIDEO
# =========================

def process_youtube_video(
    youtube_url: str
):

    global chunk_df
    global index

    # -------------------------
    # DOWNLOAD TRANSCRIPT
    # -------------------------

    ydl_opts = {

        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "skip_download": True,
        "outtmpl": os.path.join(
            DOWNLOAD_DIR,
            "%(title)s.%(ext)s"
        )

    }

    with yt_dlp.YoutubeDL(
        ydl_opts
    ) as ydl:

        info = ydl.extract_info(
            youtube_url,
            download=True
        )

        video_title = info["title"]

    # -------------------------
    # FIND VTT FILE
    # -------------------------

    vtt_file = None

    for file in os.listdir(
        DOWNLOAD_DIR
    ):

        if file.endswith(".vtt"):

            vtt_file = os.path.join(
                DOWNLOAD_DIR,
                file
            )

            break

    if vtt_file is None:

        return {

            "error":
            "Transcript not found"

        }

    # -------------------------
    # READ TRANSCRIPT
    # -------------------------

    captions = webvtt.read(
        vtt_file
    )

    full_text = ""

    for caption in captions:

        full_text += (
            caption.text + " "
        )

    full_text = (
        full_text
        .replace("\n", " ")
        .replace("  ", " ")
    )

    # -------------------------
    # CHUNKING
    # -------------------------

    words = full_text.split()

    chunk_size = 120

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(

            words[
                i:i + chunk_size
            ]

        )

        chunks.append(chunk)

    # -------------------------
    # CREATE DATAFRAME
    # -------------------------

    chunk_df = pd.DataFrame({

        "chunk_id":
        range(len(chunks)),

        "chunk":
        chunks

    })

    # -------------------------
    # CREATE EMBEDDINGS
    # -------------------------

    embeddings = model.encode(
        chunks,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    # -------------------------
    # CREATE FAISS INDEX
    # -------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    # -------------------------
    # SUCCESS RESPONSE
    # -------------------------

    return {

        "status":
        "Video processed successfully",

        "video_title":
        video_title,

        "chunks_created":
        len(chunks),

        "embeddings_created":
        True

    }

# =========================
# ASK VIDEO MENTOR
# =========================

def ask_video_mentor(
    question: str
):

    global chunk_df
    global index

    # -------------------------
    # CHECK PROCESSING
    # -------------------------

    if chunk_df is None or index is None:

        return {

            "error":
            "Process a video first"

        }

    # -------------------------
    # QUESTION EMBEDDING
    # -------------------------

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding,
        dtype=np.float32
    )

    # -------------------------
    # SEARCH VECTOR DB
    # -------------------------

    distances, indices = index.search(
        question_embedding,
        k=3
    )

    retrieved_chunks = []

    combined_context = ""

    for idx in indices[0]:

        chunk = chunk_df.iloc[idx][
            "chunk"
        ]

        retrieved_chunks.append(
            chunk
        )

        combined_context += (
            chunk + "\n\n"
        )

    # -------------------------
    # AI RESPONSE
    # -------------------------

    mentor_response = f"""
Based on the lecture:

{combined_context}

Explanation:
This topic is important in deep learning and AI systems.
Focus on understanding concepts step-by-step.

Study Tip:
Practice consistently and revise the core ideas carefully.
"""

    return {

        "question":
        question,

        "retrieved_context":
        retrieved_chunks,

        "mentor_response":
        mentor_response.strip()

    }
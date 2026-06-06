import os
import json
import faiss
import webvtt
import yt_dlp
import ollama
import pickle
import numpy as np
import pandas as pd

from sentence_transformers import (
    SentenceTransformer
)

from config.supabase_client import supabase

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
transcript_text = ""
current_session_id = None
current_user_id = None
current_video_duration = 0

# =========================
# AI SUMMARY
# =========================

def generate_video_summary():

    global transcript_text

    if not transcript_text:

        return ""

    prompt = f"""
You are an educational note generator.
Create concise revision notes ONLY from the lecture transcript.
Rules:
- Do NOT add external knowledge.
- Do NOT explain concepts not mentioned in transcript.
- Use only facts directly stated in the lecture.
- Use simple student-friendly language.
- Use headings and bullet points.
- Keep notes short and exam-focused.
- Maximum 300 words.
LECTURE:
{transcript_text[:5000]}
"""

    response = ollama.chat(

        model="qwen2.5:3b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )

    summary = response["message"]["content"]

    try:

        session_id = current_session_id

        print(
            "CURRENT SESSION ID:",
            current_session_id
        )

        if session_id is not None:

            supabase.table(
                "video_sessions"
            ).update({
                "summary": summary
            }).eq(
                "id",
                session_id
            ).execute()

        else:

            print(
                "WARNING: No active session id to update summary."
            )

    except Exception as e:

        print("ERROR updating summary in Supabase:", e)

    return summary

# =========================
# FLASHCARD GENERATOR
# =========================

def generate_video_flashcards():

    global transcript_text

    if not transcript_text:

        return []

    prompt = f"""
Return ONLY a JSON array.
Format:
[
  {{
    "front":"question",
    "back":"answer"
  }}
]
Rules:
- Generate exactly 10 flashcards.
- Use ONLY information explicitly mentioned in the lecture.
- Do NOT use external AI knowledge.
- Do NOT invent concepts.
- If a fact is not directly stated in the transcript, do NOT create a flashcard about it.
- Questions should be simple and factual.
- Answers should be under 20 words.
- Return valid JSON only.
LECTURE:
{transcript_text[:5000]}
"""

    response = ollama.chat(

        model="qwen2.5:3b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )

    raw = response["message"]["content"]

    print("\nFLASHCARD RAW RESPONSE:")
    print(raw)

    try:

        raw = raw.strip()

        # Remove markdown code blocks
        if "```json" in raw:
            raw = raw.split("```json")[1]
            raw = raw.split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1]
            raw = raw.split("```")[0]

        # Find first JSON array — strips
        # any preamble text before the [
        start = raw.find("[")
        end = raw.rfind("]")

        if start != -1 and end != -1:
            raw = raw[start:end + 1]

        flashcards = json.loads(raw)

        try:

            session_id = current_session_id

            print(
                "CURRENT SESSION ID:",
                current_session_id
            )

            if session_id is not None:

                supabase.table(
                    "video_sessions"
                ).update({
                    "flashcards": flashcards
                }).eq(
                    "id",
                    session_id
                ).execute()

            else:

                print(
                    "WARNING: No active session id to update flashcards."
                )

        except Exception as e:

            print(
                "ERROR updating flashcards in Supabase:", e
            )

        return flashcards

    except Exception as e:

        print("FLASHCARD PARSE ERROR:", e)

        return []

# =========================
# QUIZ GENERATOR
# =========================

def generate_video_quiz():

    global transcript_text

    if not transcript_text:

        return []

    prompt = f"""
Return ONLY a JSON array.
Format:
[
  {{
    "question":"",
    "options":[
      "",
      "",
      "",
      ""
    ],
    "answer":""
  }}
]
Rules:
- Generate exactly 10 MCQs.
- Use ONLY lecture content.
- Never use external knowledge.
- If the answer is not explicitly stated in the transcript, DO NOT create the question.
- Never ask about concepts not directly mentioned.
- Every question must be answerable from a direct sentence in the transcript.
- One answer must exactly match one option.
- Make questions factual.
- Avoid inference questions.
- Avoid opinion questions.
- Avoid difficult reasoning questions.
- Return valid JSON only.
LECTURE:
{transcript_text[:5000]}
"""

    response = ollama.chat(

        model="qwen2.5:3b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )

    raw = response["message"]["content"]

    print("\nQUIZ RAW RESPONSE:")
    print(raw)

    try:

        raw = raw.strip()

        # Remove markdown code blocks
        if "```json" in raw:
            raw = raw.split("```json")[1]
            raw = raw.split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1]
            raw = raw.split("```")[0]

        # Find first JSON array — strips
        # any preamble text before the [
        start = raw.find("[")
        end = raw.rfind("]")

        if start != -1 and end != -1:
            raw = raw[start:end + 1]

        quiz = json.loads(raw)

        try:

            session_id = current_session_id

            print(
                "CURRENT SESSION ID:",
                current_session_id
            )

            if session_id is not None:

                supabase.table(
                    "video_sessions"
                ).update({
                    "quiz": quiz
                }).eq(
                    "id",
                    session_id
                ).execute()

            else:

                print(
                    "WARNING: No active session id to update quiz."
                )

        except Exception as e:

            print(
                "ERROR updating quiz in Supabase:", e
            )

        return quiz

    except Exception as e:

        print("QUIZ PARSE ERROR:", e)

        return []

# =========================
# PROCESS YOUTUBE VIDEO
# =========================

def process_youtube_video(
    youtube_url: str,
    user_id: str = None
):

    global chunk_df
    global index
    global transcript_text
    global current_session_id
    global current_user_id
    global current_video_duration

    # -------------------------
    # DOWNLOAD TRANSCRIPT
    # -------------------------

    cookie_path = os.path.join(
        BASE_DIR,
        "cookies.txt"
    )

    ydl_opts = {

        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "cookiefile": cookie_path,
        "http_headers": {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0 Safari/537.36"
        },
        "outtmpl": os.path.join(
            DOWNLOAD_DIR,
            "%(title)s.%(ext)s"
        )

    }

    try:

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            info = ydl.extract_info(
                youtube_url,
                download=False
            )

            video_title = info["title"]

            video_duration = info.get(
                "duration",
                0
            )

    except Exception as e:

        print("YOUTUBE DOWNLOAD ERROR:", e)

        return {
            "error": f"YouTube extraction failed: {str(e)}"
        }

    # -------------------------
    # STORE USER + DURATION
    # -------------------------

    print(
        "USER ID RECEIVED:",
        user_id
    )

    current_user_id = user_id
    current_video_duration = video_duration

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
    # STORE TRANSCRIPT
    # -------------------------

    transcript_text = full_text

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

    print("Transcript Loaded")

    print("Creating Embeddings")

    embeddings = model.encode(
        chunks,
        show_progress_bar=False
    )

    print("Embeddings Done")

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
    # SAVE TO SUPABASE
    # -------------------------

    try:

        insert_response = (
            supabase.table("video_sessions")
            .insert(
                {
                    "user_id": user_id,
                    "video_title": video_title,
                    "youtube_url": youtube_url,
                    "watch_time": 0,
                    "completed": False,
                    "chunks_created": len(chunks)
                }
            )
            .execute()
        )

        print("SUPABASE INSERT RESPONSE:")
        print(insert_response)

        # Save the session id for this session
        if (
            insert_response.data
            and len(insert_response.data) > 0
        ):
            current_session_id = (
                insert_response.data[0]["id"]
            )

        print(
            "CURRENT SESSION ID SET:",
            current_session_id
        )

    except Exception as e:

        print(
            "ERROR inserting session into Supabase:", e
        )

    # -------------------------
    # RETURN
    # -------------------------

    return {

        "status":
        "Video processed successfully",

        "video_title":
        video_title,

        "chunks_created":
        len(chunks),

        "embeddings_created":
        True,

        "session_id":
        current_session_id

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
    # OLLAMA RESPONSE
    # -------------------------

    response = ollama.chat(

        model="mistral:latest",

        messages=[
            {
                "role": "user",
                "content": f"""
Answer ONLY using the lecture context.
LECTURE:
{combined_context}
QUESTION:
{question}
"""
            }
        ]

    )

    mentor_response = response[
        "message"
    ]["content"]

    return {

        "question":
        question,

        "retrieved_context":
        retrieved_chunks,

        "mentor_response":
        mentor_response

    }

# =========================
# UPDATE WATCH TIME
# =========================

def update_watch_time(
    session_id,
    watch_time
):

    try:

        supabase.table(
            "video_sessions"
        ).update(
            {
                "watch_time": watch_time
            }
        ).eq(
            "id",
            session_id
        ).execute()

    except Exception as e:

        print(
            "WATCH TIME ERROR:",
            e
        )

# =========================
# MARK VIDEO COMPLETED
# =========================

def mark_video_completed(
    session_id
):

    try:

        supabase.table(
            "video_sessions"
        ).update(
            {
                "completed": True
            }
        ).eq(
            "id",
            session_id
        ).execute()

    except Exception as e:

        print(
            "COMPLETION ERROR:",
            e
        )

# =========================
# GET SUMMARY
# =========================

def get_video_summary():

    return {

        "summary":
        generate_video_summary(),

        "session_id":
        current_session_id

    }

# =========================
# GET FLASHCARDS
# =========================

def get_video_flashcards():

    return {

        "flashcards":
        generate_video_flashcards(),

        "session_id":
        current_session_id

    }

# =========================
# GET QUIZ
# =========================

def get_video_quiz():

    return {

        "quiz":
        generate_video_quiz(),

        "session_id":
        current_session_id

    }

from fastapi import APIRouter
from pydantic import BaseModel

from services.video_service import (

    process_youtube_video,
    ask_video_mentor

)

router = APIRouter()

# =========================
# REQUEST MODELS
# =========================

class VideoRequest(
    BaseModel
):

    youtube_url: str

class QuestionRequest(
    BaseModel
):

    question: str

# =========================
# HOME
# =========================

@router.get("/")

def video_home():

    return {

        "module":
        "Video Intelligence Running",

        "features": [

            "YouTube Processing",
            "Transcript Extraction",
            "Chunking",
            "Embeddings",
            "FAISS Retrieval",
            "Video RAG Mentor"

        ]

    }

# =========================
# PROCESS VIDEO
# =========================

@router.post("/process")

def process_video(
    request: VideoRequest
):

    return process_youtube_video(
        request.youtube_url
    )

# =========================
# ASK VIDEO MENTOR
# =========================

@router.post("/ask")

def ask_video_question(
    request: QuestionRequest
):

    return ask_video_mentor(
        request.question
    )
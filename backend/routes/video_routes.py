from fastapi import APIRouter
from pydantic import BaseModel

from services.video_service import (
    process_youtube_video,
    ask_video_mentor,
    get_video_summary,
    get_video_flashcards,
    get_video_quiz,
    update_watch_time,
    mark_video_completed
)

router = APIRouter()

# =========================
# REQUEST MODELS
# =========================

class VideoRequest(BaseModel):
    youtube_url: str
    user_id: str = None


class QuestionRequest(BaseModel):
    question: str


class WatchTimeRequest(BaseModel):
    session_id: int
    watch_time: int


# =========================
# HOME
# =========================

@router.get("/")
def video_home():

    return {

        "status": "success",

        "module": "Video Intelligence Running",

        "features": [

            "YouTube Processing",
            "Transcript Extraction",
            "Chunking",
            "Embeddings",
            "FAISS Retrieval",
            "Video RAG Mentor",
            "AI Summary",
            "AI Flashcards",
            "AI Quiz",
            "Watch Time Tracking",
            "Video Completion"

        ]
    }


# =========================
# PROCESS VIDEO
# =========================

@router.post("/process")
def process_video(
    request: VideoRequest
):

    try:

        return process_youtube_video(
            request.youtube_url,
            request.user_id
        )

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)

        }


# =========================
# ASK VIDEO MENTOR
# =========================

@router.post("/ask")
def ask_video_question(
    request: QuestionRequest
):

    try:

        return ask_video_mentor(
            request.question
        )

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)

        }


# =========================
# GENERATE SUMMARY
# =========================

@router.get("/summary")
def generate_summary():

    try:

        return get_video_summary()

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)

        }


# =========================
# GENERATE FLASHCARDS
# =========================

@router.get("/flashcards")
def generate_flashcards():

    try:

        return get_video_flashcards()

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)

        }


# =========================
# GENERATE QUIZ
# =========================

@router.get("/quiz")
def generate_quiz():

    try:

        return get_video_quiz()

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)

        }


# =========================
# UPDATE WATCH TIME
# =========================

@router.post("/watch-time")
def update_watch_time_route(
    request: WatchTimeRequest
):

    try:

        update_watch_time(
            request.session_id,
            request.watch_time
        )

        return {
            "status": "success"
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)

        }


# =========================
# MARK VIDEO COMPLETE
# =========================

@router.post("/complete")
def complete_video_route(
    request: WatchTimeRequest
):

    try:

        mark_video_completed(
            request.session_id
        )

        return {
            "status": "success"
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)

        }

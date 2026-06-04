from fastapi import APIRouter
from pydantic import BaseModel

from services.quiz_service import (

    generate_quiz,
    get_revision_notes,
    get_flashcards

)

router = APIRouter()

# =========================
# HOME ROUTE
# =========================

@router.get("/")

def quiz_home():

    return {

        "module":
        "Quiz + Flashcards Running",

        "features": [

            "MCQ Quiz Generator",
            "Flashcards",
            "Revision Notes",
            "Transcript Intelligence"

        ]

    }

# =========================
# REQUEST MODEL
# =========================

class QuizRequest(BaseModel):

    topic: str

# =========================
# MCQ QUIZ GENERATOR
# =========================

@router.post("/mcqs")

def mcq_quiz(request: QuizRequest):

    return generate_quiz(
        request.topic
    )

# =========================
# REVISION NOTES
# =========================

@router.post("/revision-notes")

def revision_notes(request: QuizRequest):

    return get_revision_notes(
        request.topic
    )

# =========================
# FLASHCARDS
# =========================

@router.post("/flashcards")

def flashcards(request: QuizRequest):

    return get_flashcards(
        request.topic
    )
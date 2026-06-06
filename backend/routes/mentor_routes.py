from fastapi import APIRouter

from schemas.mentor_schema import (
    MentorQuestion
)

from services.mentor_service import (
    ask_mentor
)

router = APIRouter()

# =========================
# AI MENTOR CHAT
# =========================

@router.post("/ask")
def mentor_chat(
    request: MentorQuestion
):

    response = ask_mentor(

        request.user_id,

        request.question

    )

    return response


# =========================
# MENTOR STATUS
# =========================

@router.get("/")
def mentor_status():

    return {

        "module":
        "AI Mentor Running Successfully",

        "features": [

            "RAG Retrieval",

            "Transcript Embeddings",

            "Semantic Search",

            "Persona Intelligence",

            "Academic Risk Awareness",

            "Cognitive Intelligence",

            "Chat History Memory",

            "Ollama Llama3 Integration",

            "Personalized Guidance"

        ]

    }
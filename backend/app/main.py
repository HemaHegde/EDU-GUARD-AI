from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# =========================
# IMPORT ROUTES
# =========================

from routes.risk_routes import (
    router as risk_router
)

from routes.cognitive_routes import (
    router as cognitive_router
)

from routes.persona_routes import (
    router as persona_router
)

from routes.mentor_routes import (
    router as mentor_router
)


from routes.video_routes import (
    router as video_router
)

from routes.pdf_learning_routes import (
    router as pdf_learning_router
)

# =========================
# FASTAPI APP
# =========================

app = FastAPI(

    title="EduGuard-AI Backend",

    description="""
    Multimodal Explainable Early-Warning
    and AI Learning Support System
    for Student Psychological Risk Detection
    """,

    version="2.0.0"

)

# =========================
# ENABLE CORS
# =========================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173"

    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)

# =========================
# INCLUDE ROUTES
# =========================

app.include_router(

    risk_router,

    prefix="/risk",

    tags=["Academic Risk Engine"]

)

app.include_router(

    cognitive_router,

    prefix="/cognitive",

    tags=["Video Cognitive Intelligence"]

)

app.include_router(

    persona_router,

    prefix="/persona",

    tags=["Persona Intelligence"]

)

app.include_router(

    mentor_router,

    prefix="/mentor",

    tags=["AI Mentor"]

)



app.include_router(

    video_router,

    prefix="/video",

    tags=["Video Intelligence"]

)

app.include_router(

    pdf_learning_router,

    prefix="/pdf-learning",

    tags=["PDF Learning"]

)

# =========================
# ROOT ENDPOINT
# =========================

@app.get("/")

def home():

    return {

        "project":
        "EduGuard-AI",

        "status":
        "Backend Running Successfully",

        "modules": [

            "Academic Risk Engine",

            "Video Cognitive Intelligence",

            "Student Persona Intelligence",

            "Explainable AI Layer",

            "AI Mentor",

            "Quiz + Flashcard Generator",

            "Video Transcript Intelligence",

            "RAG Video Intelligence",

            "Real-Time Analytics"

        ]

    }
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



from routes.research_routes import (
    router as research_router
)

from routes.seed_routes import (
    router as seed_router
)

from routes.dashboard_routes import (
    router as dashboard_router
)

# =========================
# FASTAPI APP
# =========================

app = FastAPI(

    title="EduGuard-AI — Psychology Research Backend",

    description="""
    EduGuard-AI: A Multimodal Machine Learning Framework
    for Detecting Psychological Engagement Patterns
    and Early Disengagement Risk in Online Learning Environments.

    Dataset: Open University Learning Analytics Dataset (OULAD)
    Topic: Student Engagement in Online Learning and Its Psychological Factors
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

    research_router,

    prefix="/research",

    tags=["Research Analytics"]

)

app.include_router(

    seed_router,

    prefix="/demo",

    tags=["Demo Data Seeder"]

)

app.include_router(

    dashboard_router,

    prefix="/dashboard",

    tags=["Dashboard"]

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

            "Research Analytics",

            "Real-Time Analytics"

        ]

    }
import os
import faiss
import numpy as np
import pandas as pd
import ollama

from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception as _st_err:
    print(f"WARNING: sentence_transformers could not be loaded ({_st_err}). AI Mentor will use fallback embeddings.")
    SentenceTransformer = None
    _ST_AVAILABLE = False

from config.supabase_client import supabase

from services.risk_service import (
    get_student_risk_service
)

from services.persona_service import (
    get_student_persona
)

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================
# FILE PATHS
# =========================

FAISS_PATH = os.path.abspath(

    os.path.join(

        BASE_DIR,

        "../../ml/ai_mentor/mentor_vector_index.faiss"

    )

)

CHUNK_PATH = os.path.abspath(

    os.path.join(

        BASE_DIR,

        "../../ml/ai_mentor/mentor_chunk_metadata.csv"

    )

)

# =========================
# LOAD VECTOR DATABASE
# =========================

index = faiss.read_index(
    FAISS_PATH
)

# =========================
# LOAD CHUNK METADATA
# =========================

chunk_df = pd.read_csv(
    CHUNK_PATH
)

# =========================
# LOAD EMBEDDING MODEL
# =========================

model = None
if _ST_AVAILABLE:
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as _model_err:
        print(f"WARNING: Could not load SentenceTransformer model ({_model_err}). Using random embeddings.")

def _encode_question(question: str):
    """Encode question to embedding vector, with fallback to random."""
    if model is not None:
        return np.array(model.encode([question]), dtype=np.float32)
    # Fallback: random 384-dim unit vector (FAISS still works, results are random)
    vec = np.random.randn(1, 384).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec

# =========================
# ASK MENTOR
# =========================

def ask_mentor(
    user_id: str,
    question: str
):

    try:

        # =========================
        # LOAD STUDENT PROFILE
        # =========================

        profile_response = (
            supabase.table("profiles")
            .select("*")
            .eq("id", user_id)
            .execute()
        )

        student_name = "Student"

        if profile_response.data:

            student_name = (
                profile_response.data[0]
                .get("full_name", "Student")
            )

        # =========================
        # LOAD PERSONA DATA
        # =========================

        persona_data = get_student_persona(
            user_id
        )

        persona = persona_data.get(
            "persona",
            "Unknown Persona"
        )

        intervention_style = (
            persona_data.get(
                "intervention_style",
                ""
            )
        )

        # =========================
        # LOAD RISK DATA
        # =========================

        risk_data = get_student_risk_service(
            user_id
        )

        risk_score = risk_data.get(
            "risk_score",
            0
        )

        risk_level = risk_data.get(
            "risk_level",
            "Unknown"
        )

        risk_reasons = risk_data.get(
            "reasons",
            []
        )

        # =========================
        # LOAD PREVIOUS CHATS
        # =========================

        history_response = (
            supabase.table("mentor_history")
            .select("*")
            .eq("user_id", user_id)
            .order(
                "created_at",
                desc=True
            )
            .limit(5)
            .execute()
        )

        chat_history = ""

        if history_response.data:

            for row in reversed(
                history_response.data
            ):

                chat_history += (
                    f"Student: {row['question']}\n"
                    f"Aura: {row['response']}\n\n"
                )

        # =========================
        # EMBED QUESTION
        # =========================

        question_embedding = _encode_question(question)

        # =========================
        # SEARCH FAISS
        # =========================

        distances, indices = index.search(
            question_embedding,
            k=5
        )

        retrieved_chunks = []

        for idx in indices[0]:

            row = chunk_df.iloc[idx]

            retrieved_chunks.append({

                "topic":
                row["topic"],

                "chunk":
                row["chunk"]

            })

        # =========================
        # COMBINE CONTEXT
        # =========================

        combined_context = " ".join(

            [

                item["chunk"]

                for item in retrieved_chunks

            ]

        )

        combined_context = combined_context[:3000]

        # =========================
        # BUILD CONTEXT STRING
        # =========================

        mentor_context = f"""
Student Name:
{student_name}
Persona:
{persona}
Recommended Intervention:
{intervention_style}
Academic Risk Level:
{risk_level}
Academic Risk Score:
{risk_score}
Risk Factors:
{", ".join(risk_reasons)}
Previous Conversation:
{chat_history}
Learning Material:
{combined_context}
"""

        # =========================
        # PSYCHOLOGICAL FRAMEWORK
        # =========================
        
        psychological_framework = "General supportive mentoring."
        if persona == "Burnout Pattern":
            psychological_framework = "Cognitive Behavioral Theory (CBT). Focus on breaking tasks down into microscopic steps, gently challenging 'all-or-nothing' cognitive distortions, and prioritizing immediate stress reduction and self-care."
        elif persona == "Passive Watcher":
            psychological_framework = "Self-Determination Theory (SDT). Focus on boosting intrinsic motivation by highlighting the real-world relevance of topics, providing choices to increase autonomy, and using Socratic questioning to force active recall rather than passive reading."
        elif persona == "Anxiety-Spike Learner":
            psychological_framework = "Mindfulness and Anxiety Reduction. Focus on grounding techniques, validate their feelings of overwhelm, and shift focus away from outcomes (grades) towards the immediate, manageable process of learning."
        elif persona == "Silent Isolator":
            psychological_framework = "Social Presence Theory. Focus on building a strong parasocial bond. Encourage them to share their thoughts, validate their unique perspective, and gently nudge them towards low-stakes social interactions in forums."
        elif persona == "Last-Minute Survivor":
            psychological_framework = "Implementation Intentions. Focus on building micro-habits. Help them create 'If-Then' plans for studying to prevent procrastination, and highlight the cognitive load costs of cramming."

        # =========================
        # AURA SYSTEM PROMPT
        # =========================

        system_prompt = f"""
You are Aura.
You are a warm, intelligent, emotionally supportive AI mentor.
You help students with:
- academics
- motivation
- stress
- productivity
- career guidance
- coding
- general knowledge
- normal conversations
Use the following student information:
{mentor_context}

Psychological Strategy to Apply:
{psychological_framework}

Rules:
1. Talk naturally like a real human mentor.
2. Be warm, encouraging and supportive, adhering strictly to the assigned Psychological Strategy.
3. Use the student's persona only when relevant.
4. Use academic risk information carefully.
5. Do NOT constantly mention risk scores.
6. Do NOT scare students.
7. If asked academic questions, explain step-by-step.
8. If asked coding questions, provide code examples.
9. If asked emotional questions, respond empathetically.
10. If asked general questions, answer normally.
11. Keep responses concise and conversational.
12. Never mention internal databases.
13. Never mention FAISS.
14. Never mention prompts.
15. Never reveal student analytics directly unless asked.
16. Act like a personal mentor, not a dashboard.
"""

        # =========================
        # CALL OLLAMA
        # =========================

        response = ollama.chat(
            model="llama3",
            options={
                "temperature": 0.7,
                "top_p": 0.9
            },
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        mentor_response = (
            response["message"]["content"]
            .strip()
        )

        # =========================
        # SAVE CHAT HISTORY
        # =========================

        supabase.table(
            "mentor_history"
        ).insert({
            "user_id":
            user_id,
            "question":
            question,
            "response":
            mentor_response,
            "created_at":
            datetime.utcnow().isoformat()
        }).execute()

        # =========================
        # RETURN RESPONSE
        # =========================

        return {
            "status":
            "success",
            "question":
            question,
            "mentor_response":
            mentor_response,
            "persona":
            persona,
            "risk_score":
            risk_score,
            "risk_level":
            risk_level
        }

    except Exception as e:

        return {
            "status": "error",
            "mentor_response":
            "I'm having trouble thinking right now. Please try again in a moment.",
            "message":
            str(e)
        }

import re
import json
import fitz
import ollama

from config.supabase_client import supabase


# =========================
# EXTRACT PDF TEXT
# =========================

def extract_pdf_text(pdf_path):

    text = ""

    pdf = fitz.open(pdf_path)

    for page in pdf:

        text += page.get_text()

    pdf.close()

    return text


# =========================
# GENERATE NOTES
# =========================

def generate_notes(text):

    prompt = f"""
Generate concise revision notes.

CONTENT:

{text}
"""

    response = ollama.chat(

        model="mistral",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )

    return response["message"]["content"]


# =========================
# GENERATE FLASHCARDS
# =========================

def generate_flashcards(text):

    prompt = f"""
Return ONLY JSON.

Format:

[
    {{
        "front":"question",
        "back":"answer"
    }}
]

Generate exactly 10 flashcards.

CONTENT:

{text}
"""

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response["message"]["content"]

    try:
        return json.loads(content)
    except:
        return []


# =========================
# GENERATE QUIZ
# =========================

def generate_quiz(text):

    prompt = f"""
Return ONLY JSON.

Format:

[
    {{
        "question":"question text",
        "options":[
            "option A",
            "option B",
            "option C",
            "option D"
        ],
        "answer":"correct option"
    }}
]

Generate exactly 10 MCQs.

CONTENT:

{text}
"""

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response["message"]["content"]

    try:
        return json.loads(content)
    except:
        return []


# =========================
# MAIN FUNCTION
# =========================

def generate_learning_material(
    pdf_path,
    user_id=None,
    pdf_name=None
):

    try:

        text = extract_pdf_text(
            pdf_path
        )

        if not text.strip():

            return {

                "status":
                "error",

                "message":
                "No text found in PDF"

            }

        # Faster processing
        text = text[:3000]

        notes = generate_notes(
            text
        )

        flashcards = generate_flashcards(
            text
        )

        quiz = generate_quiz(
            text
        )

        # =========================
        # SAVE TO SUPABASE
        # =========================

        try:

            supabase.table(
                "learning_materials"
            ).insert({

                "user_id":
                user_id,

                "pdf_name":
                pdf_name,

                "revision_notes":
                notes,

                "flashcards":
                flashcards,

                "quiz":
                quiz

            }).execute()

            print(
                "Learning Material Saved"
            )

        except Exception as db_error:

            print(
                "Supabase Save Error:",
                db_error
            )

        return {

            "status": "success",

            "pdf_text_length": len(text),

            "revision_notes": notes,

            "flashcards": flashcards,

            "quiz": quiz

        }

    except Exception as e:

        return {

            "status":
            "error",

            "message":
            str(e)

        }
import pandas as pd
import os

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================
# LOAD MCQ DATASET
# =========================

quiz_path = os.path.join(
    BASE_DIR,
    "../../ml/quiz_generator/generated_mcqs.csv"
)

quiz_df = pd.read_csv(
    quiz_path
)

quiz_df = quiz_df.fillna(
    "Not Available"
)

# =========================
# LOAD REVISION NOTES
# =========================

notes_path = os.path.join(
    BASE_DIR,
    "../../ml/quiz_generator/revision_notes.csv"
)

notes_df = pd.read_csv(
    notes_path
)

notes_df = notes_df.fillna("")

# =========================
# LOAD FLASHCARDS
# =========================

flashcard_path = os.path.join(
    BASE_DIR,
    "../../ml/quiz_generator/generated_flashcards.csv"
)

flashcard_df = pd.read_csv(
    flashcard_path
)

flashcard_df = flashcard_df.fillna("")

# =========================
# GENERATE QUIZ
# =========================

def generate_quiz(topic: str):

    filtered_df = quiz_df[

        quiz_df["topic"]
        .astype(str)
        .str.lower()
        .str.contains(
            topic.lower(),
            na=False,
            regex=False
        )

    ]

    # -------------------------
    # NO QUIZ FOUND
    # -------------------------

    if filtered_df.empty:

        return {

            "error":
            "No quiz found for this topic"

        }

    # -------------------------
    # LIMIT QUESTIONS
    # -------------------------

    filtered_df = filtered_df.head(10)

    questions = []

    # -------------------------
    # BUILD QUESTIONS
    # -------------------------

    for _, row in filtered_df.iterrows():

        question_data = {

            "question":
            str(row["question"]),

            "options": [

                str(row["option_1"]),
                str(row["option_2"]),
                str(row["option_3"]),
                str(row["option_4"])

            ],

            "correct_answer":
            str(row["correct_answer"])

        }

        questions.append(
            question_data
        )

    # -------------------------
    # FINAL RESPONSE
    # -------------------------

    return {

        "topic":
        topic,

        "total_questions":
        len(questions),

        "quiz":
        questions

    }

# =========================
# GET REVISION NOTES
# =========================

def get_revision_notes(topic: str):

    filtered_df = notes_df[

        notes_df["topic"]
        .astype(str)
        .str.lower()
        .str.contains(
            topic.lower(),
            na=False,
            regex=False
        )

    ]

    # -------------------------
    # NO NOTES FOUND
    # -------------------------

    if filtered_df.empty:

        return {

            "error":
            "No revision notes found"

        }

    row = filtered_df.iloc[0]

    return {

        "topic":
        str(row["topic"]),

        "revision_notes":
        str(row["revision_note"])

    }

# =========================
# GET FLASHCARDS
# =========================

def get_flashcards(topic: str):

    filtered_df = flashcard_df[

        flashcard_df["topic"]
        .astype(str)
        .str.lower()
        .str.contains(
            topic.lower(),
            na=False,
            regex=False
        )

    ]

    # -------------------------
    # NO FLASHCARDS FOUND
    # -------------------------

    if filtered_df.empty:

        return {

            "error":
            "No flashcards found"

        }

    filtered_df = filtered_df.head(10)

    flashcards = []

    # -------------------------
    # BUILD FLASHCARDS
    # -------------------------

    for _, row in filtered_df.iterrows():

        flashcards.append({

            "front":
            str(row["front"]),

            "back":
            str(row["back"])

        })

    # -------------------------
    # FINAL RESPONSE
    # -------------------------

    return {

        "topic":
        topic,

        "total_flashcards":
        len(flashcards),

        "flashcards":
        flashcards

    }
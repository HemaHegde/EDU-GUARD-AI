import pandas as pd
import random
import re

print("Starting Quiz + Flashcard Intelligence Engine...")

# =========================
# LOAD TRANSCRIPT CHUNKS
# =========================

chunk_df = pd.read_csv(
    "../ai_mentor/mentor_chunk_metadata.csv"
)

print("\nTranscript Chunks Loaded!")

print("\nDataset Shape:")
print(chunk_df.shape)

print("\nPreview:")
print(chunk_df.head())

# =========================
# QUESTION TEMPLATES
# =========================

question_templates = [

    "What is {}?",

    "Explain {}.",

    "Which statement best describes {}?",

    "What is the purpose of {}?",

    "How does {} work?"
]

# =========================
# CONCEPT EXTRACTION
# =========================

print("\nExtracting learning concepts...")

def extract_concepts(text):

    words = text.split()

    concepts = []

    for word in words:

        if len(word) > 6:

            concepts.append(word)

    return list(set(concepts))

# =========================
# GENERATE QUIZZES
# =========================

print("\nGenerating MCQs...")

quiz_data = []

for idx, row in chunk_df.iterrows():

    topic = row["topic"]

    chunk = row["chunk"]

    concepts = extract_concepts(chunk)

    concepts = concepts[:5]

    for concept in concepts:

        question = random.choice(
            question_templates
        ).format(concept)

        correct_answer = concept

        distractors = random.sample(
            concepts,
            min(
                len(concepts),
                3
            )
        )

        options = list(
            set(
                distractors + [correct_answer]
            )
        )

        random.shuffle(options)

        quiz_data.append({

            "topic": topic,

            "question": question,

            "correct_answer": correct_answer,

            "option_1":
            options[0]
            if len(options) > 0 else "",

            "option_2":
            options[1]
            if len(options) > 1 else "",

            "option_3":
            options[2]
            if len(options) > 2 else "",

            "option_4":
            options[3]
            if len(options) > 3 else ""

        })

quiz_df = pd.DataFrame(
    quiz_data
)

print("\nQuiz Dataset Preview:")
print(quiz_df.head())

print("\nTotal MCQs Generated:")
print(len(quiz_df))

# =========================
# GENERATE FLASHCARDS
# =========================

print("\nGenerating flashcards...")

flashcards = []

for idx, row in chunk_df.iterrows():

    topic = row["topic"]

    chunk = row["chunk"]

    concepts = extract_concepts(chunk)

    concepts = concepts[:5]

    for concept in concepts:

        flashcards.append({

            "topic": topic,

            "front":
            f"What is {concept}?",

            "back":
            f"{concept} is an important concept in {topic}."

        })

flashcard_df = pd.DataFrame(
    flashcards
)

print("\nFlashcard Preview:")
print(flashcard_df.head())

print("\nTotal Flashcards Generated:")
print(len(flashcard_df))

# =========================
# GENERATE REVISION NOTES
# =========================

print("\nGenerating revision notes...")

revision_notes = []

for idx, row in chunk_df.iterrows():

    topic = row["topic"]

    chunk = row["chunk"]

    sentences = re.split(
        r'(?<=[.!?]) +',
        chunk
    )

    summary = " ".join(
        sentences[:2]
    )

    revision_notes.append({

        "topic": topic,

        "revision_note": summary

    })

revision_df = pd.DataFrame(
    revision_notes
)

print("\nRevision Notes Preview:")
print(revision_df.head())

# =========================
# SAVE OUTPUTS
# =========================

quiz_df.to_csv(
    "generated_mcqs.csv",
    index=False
)

flashcard_df.to_csv(
    "generated_flashcards.csv",
    index=False
)

revision_df.to_csv(
    "revision_notes.csv",
    index=False
)

print("\nAll educational assets saved successfully!")

print("\nQuiz + Flashcard Intelligence Completed!")
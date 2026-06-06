import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useState } from "react";
import {
  BookOpen,
  FileText,
  RotateCw,
} from "lucide-react";

import { PageShell } from "@/components/eg/PageShell";

export const Route = createFileRoute("/learn")({
  head: () => ({
    meta: [
      {
        title: "Quiz & Flashcards · EduGuard-AI",
      },
    ],
  }),

  component: LearnPage,
});

// =========================
// COMPONENT
// =========================

function LearnPage() {
  const [pdfFile, setPdfFile] =
    useState<File | null>(null);

  const [quiz, setQuiz] =
    useState<any[]>([]);

  const [flashcards, setFlashcards] =
    useState<any[]>([]);

  const [notes, setNotes] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [answers, setAnswers] =
    useState<Record<number, string>>({});

  const [submitted, setSubmitted] =
    useState(false);

  const [score, setScore] =
    useState(0);

  const USER_ID =
    "d271bd7d-6631-4969-b000-692ad069ddfe";

  // =========================
  // UPLOAD PDF
  // =========================

  async function uploadPDF() {

    if (!pdfFile) {

      alert("Please select PDF");

      return;
    }

    try {

      setError("");

      setAnswers({});

      setSubmitted(false);

      setScore(0);

      setLoading(true);

      const formData =
        new FormData();

      formData.append(
        "file",
        pdfFile
      );

      const response =
        await fetch(

          "http://127.0.0.1:8000/pdf-learning/upload",

          {
            method: "POST",
            body: formData,
          }

        );

      const data =
        await response.json();

      console.log(
        "BACKEND RESPONSE:",
        data
      );

      if (
        data.status !==
        "success"
      ) {

        throw new Error(
          data.message ||
          "Generation Failed"
        );
      }

      setNotes(

        data.result
          ?.revision_notes || ""

      );

      setFlashcards(

        data.result
          ?.flashcards || []

      );

      setQuiz(

        data.result
          ?.quiz || []

      );

    } catch (err: any) {

      setError(
        err.message
      );

      console.error(err);

    } finally {

      setLoading(false);
    }
  }

  // =========================
  // SUBMIT QUIZ
  // =========================

  async function submitQuiz() {

    let finalScore = 0;

    quiz.forEach((q, index) => {

      if (
        answers[index] ===
        q.answer
      ) {

        finalScore++;

      }

    });

    setScore(
      finalScore
    );

    setSubmitted(
      true
    );

    try {

      await fetch(

        "http://127.0.0.1:8000/pdf-learning/save-attempt",

        {

          method: "POST",

          headers: {

            "Content-Type":
            "application/json"

          },

          body: JSON.stringify({

            user_id:
            USER_ID,

            pdf_name:
            pdfFile?.name ||

            "Unknown PDF",

            score:
            finalScore,

            total:
            quiz.length

          })

        }

      );

    }

    catch (err) {

      console.error(
        err
      );

    }

  }

  // =========================
  // UI
  // =========================

  return (
    <PageShell
      title="PDF Learning Intelligence"
      description="Upload study material and automatically generate revision notes, flashcards and MCQ quizzes."
    >
      {/* ========================= */}
      {/* FILE UPLOAD */}
      {/* ========================= */}

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card rounded-3xl p-6"
      >
        <div className="mt-5 flex gap-3">
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                setPdfFile(e.target.files[0]);
              }
            }}
            className="flex-1 rounded-2xl border border-border bg-white/70 px-4 py-3"
          />

          <button
            onClick={uploadPDF}
            className="rounded-2xl bg-primary px-6 py-3 text-primary-foreground"
          >
            Generate Learning Material
          </button>
        </div>

        {error && (
          <p className="mt-3 text-sm text-red-500">{error}</p>
        )}
      </motion.div>

      {/* ========================= */}
      {/* QUIZ */}
      {/* ========================= */}

      <section>
        <SectionHeader
          icon={<BookOpen className="h-4 w-4" />}
          title="Quiz"
        />

        {loading ? (
          <LoadingCard />
        ) : quiz.length > 0 ? (
          <div className="space-y-4">
            {quiz.map((q, index) => (
              <div
                key={index}
                className="glass-card rounded-3xl p-5"
              >
                <h3 className="mb-4 font-semibold">
                  {index + 1}. {q.question}
                </h3>
                <div className="grid gap-2">
                  {q.options.map(
                    (
                      option: string,
                      i: number
                    ) => (
                      <button

                        key={i}

                        onClick={() =>
                          !submitted &&

                          setAnswers({

                            ...answers,

                            [index]: option

                          })

                        }

                        className={`

                          rounded-xl

                          border

                          p-3

                          text-left

                          transition

                          ${

                            submitted &&

                            option === q.answer

                            ? "bg-green-100 border-green-500"

                            :

                            submitted &&

                            answers[index] === option &&

                            option !== q.answer

                            ? "bg-red-100 border-red-500"

                            :

                            answers[index] === option

                            ? "bg-primary/10 border-primary"

                            :

                            ""

                          }

                        `}

                      >

                        {option}

                      </button>
                    )
                  )}
                </div>
              </div>
            ))}

            {/* STEP 5: Score card */}
            {submitted && (

              <div
                className="
                glass-card
                rounded-3xl
                p-5
                mt-4
                font-semibold
                "
              >

                Score: {score}/{quiz.length}

                {/* STEP 9: Percentage */}
                <div className="mt-2">
                  Percentage:{" "}

                  {Math.round(
                    (score / quiz.length) * 100
                  )}%
                </div>

              </div>

            )}

            {/* STEP 6: Submit + STEP 8: Retry */}
            <div className="mt-4">

              <button

                onClick={submitQuiz}

                disabled={
                  submitted ||
                  // STEP 12: Disable until all answered
                  Object.keys(answers).length
                  !== quiz.length
                }

                className="
                  rounded-2xl
                  bg-primary
                  px-6
                  py-3
                  text-primary-foreground
                  disabled:opacity-50
                "

              >

                Submit Quiz

              </button>

              {/* STEP 8: Retry button */}
              {submitted && (
                <button
                  onClick={() => {
                    setAnswers({});
                    setSubmitted(false);
                    setScore(0);
                  }}
                  className="
                    ml-3
                    rounded-2xl
                    border
                    px-6
                    py-3
                  "
                >
                  Retry Quiz
                </button>
              )}

            </div>

          </div>
        ) : (
          <EmptyCard title="Upload PDF First" />
        )}
      </section>

      {/* ========================= */}
      {/* FLASHCARDS */}
      {/* ========================= */}

      <section>
        <SectionHeader
          icon={<RotateCw className="h-4 w-4" />}
          title="Flashcards"
        />

        {loading ? (
          <LoadingCard />
        ) : flashcards.length > 0 ? (
          <div className="grid gap-4 md:grid-cols-2">
            {flashcards.map(
              (card, index) => (
                // STEP 11: Flip animation
                <FlashCard
                  key={index}
                  card={card}
                />
              )
            )}
          </div>
        ) : (
          <EmptyCard title="Upload PDF First" />
        )}
      </section>

      {/* ========================= */}
      {/* NOTES */}
      {/* ========================= */}

      <section>
        <SectionHeader
          icon={<FileText className="h-4 w-4" />}
          title="Revision Notes"
        />

        {loading ? (
          <LoadingCard />
        ) : notes ? (
          <div className="glass-card rounded-3xl p-6 whitespace-pre-wrap">
            {notes}
          </div>
        ) : (
          <EmptyCard title="Upload PDF First" />
        )}
      </section>
    </PageShell>
  );
}

// =========================
// FLASHCARD (STEP 11)
// =========================

function FlashCard({ card }: { card: any }) {
  const [flipped, setFlipped] =
    useState(false);

  return (
    <div
      onClick={() =>
        setFlipped(!flipped)
      }
      className="glass-card rounded-3xl p-5 cursor-pointer select-none transition"
    >
      <h3 className="font-semibold">
        {flipped ? card.back : card.front}
      </h3>
      <p className="mt-3 text-xs text-muted-foreground">
        {flipped ? "Click to see question" : "Click to reveal answer"}
      </p>
    </div>
  );
}

// =========================
// SECTION HEADER
// =========================

function SectionHeader({
  icon,
  title,
}: {
  icon: React.ReactNode;
  title: string;
}) {
  return (
    <div className="mb-4 mt-8 flex items-center gap-2">
      <span className="grid h-8 w-8 place-items-center rounded-xl bg-primary/15 text-primary">
        {icon}
      </span>
      <h2 className="text-xl font-semibold">{title}</h2>
    </div>
  );
}

// =========================
// LOADING
// =========================

function LoadingCard() {
  return (
    <div className="glass-card rounded-3xl p-10 text-center">
      Loading...
    </div>
  );
}

// =========================
// EMPTY
// =========================

function EmptyCard({ title }: { title: string }) {
  return (
    <div className="glass-card rounded-3xl p-10 text-center text-muted-foreground">
      {title}
    </div>
  );
}

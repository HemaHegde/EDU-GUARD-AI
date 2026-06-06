import { createFileRoute } from "@tanstack/react-router";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

import {
  Youtube,
  Loader2,
  Sparkles,
  FileText,
  Layers,
  Search,
  BookOpen,
  Brain,
  MessageCircle,
  CheckCircle2,
  Hash,
  Zap,
  RotateCcw,
  Trophy,
  User,
  Bot,
  ChevronLeft,
  ChevronRight,
  Send,
} from "lucide-react";

import { PageShell } from "@/components/eg/PageShell";
import { supabase } from "@/lib/supabase";

export const Route = createFileRoute("/video")({
  head: () => ({
    meta: [
      {
        title: "Video Intelligence · EduGuard-AI",
      },
    ],
  }),
  component: VideoPage,
});

// =========================
// STAGES
// =========================

const STAGES = [
  { key: "transcript", label: "Transcript", icon: FileText },
  { key: "chunking", label: "Chunking", icon: Layers },
  { key: "embeddings", label: "Embeddings", icon: Sparkles },
  { key: "semantic", label: "Semantic Search", icon: Search },
  { key: "mentor", label: "AI Mentor", icon: Brain },
  { key: "quiz", label: "Quiz + Cards", icon: BookOpen },
];

// =========================
// HELPERS
// =========================

function getVideoId(url: string) {
  const regExp = /(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&?#]+)/;
  const match = url.match(regExp);
  return match ? match[1] : "";
}

// =========================
// FLIP CARD
// =========================

function FlipCard({ card, index }: { card: any; index: number }) {
  const [flipped, setFlipped] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="h-52 cursor-pointer group"
      style={{ perspective: "1200px" }}
      onClick={() => setFlipped((f) => !f)}
      whileHover={{ scale: 1.02, y: -2 }}
    >
      <motion.div
        animate={{ rotateY: flipped ? 180 : 0 }}
        transition={{ duration: 0.55, type: "spring", stiffness: 90, damping: 14 }}
        style={{ transformStyle: "preserve-3d", position: "relative", width: "100%", height: "100%" }}
      >
        {/* FRONT */}
        <div
          style={{ backfaceVisibility: "hidden" }}
          className="absolute inset-0 flex flex-col items-center justify-center rounded-2xl border border-primary/20 bg-gradient-to-br from-white/80 via-primary/5 to-primary/10 p-5 text-center shadow-md shadow-primary/5 backdrop-blur-sm transition-shadow group-hover:shadow-lg group-hover:shadow-primary/10"
        >
          <div className="mb-2 flex items-center gap-1.5">
            <div className="h-1.5 w-1.5 rounded-full bg-primary/40" />
            <div className="text-xs font-semibold uppercase tracking-widest text-primary/50">Question</div>
            <div className="h-1.5 w-1.5 rounded-full bg-primary/40" />
          </div>
          <div className="text-sm font-medium leading-relaxed text-foreground">{card.front}</div>
          <div className="mt-4 flex items-center gap-1.5 rounded-full bg-primary/8 px-3 py-1 text-xs text-primary/60">
            <RotateCcw className="h-3 w-3" /> Tap to reveal
          </div>
        </div>

        {/* BACK */}
        <div
          style={{ backfaceVisibility: "hidden", transform: "rotateY(180deg)" }}
          className="absolute inset-0 flex flex-col items-center justify-center rounded-2xl border border-emerald-300/50 bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 p-5 text-center shadow-md shadow-emerald-100 backdrop-blur-sm"
        >
          <div className="mb-2 flex items-center gap-1.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
            <div className="text-xs font-semibold uppercase tracking-widest text-emerald-600/70">Answer</div>
          </div>
          <div className="text-sm font-medium leading-relaxed text-emerald-900">{card.back}</div>
        </div>
      </motion.div>
    </motion.div>
  );
}

// =========================
// QUIZ QUESTION
// =========================

function QuizQuestion({
  q,
  index,
  onAnswer,
  answered,
  selectedOption,
}: {
  q: any;
  index: number;
  onAnswer: (qi: number, option: string) => void;
  answered: boolean;
  selectedOption: string | null;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className="rounded-2xl border border-border/60 bg-white/70 p-5 shadow-sm backdrop-blur-sm"
    >
      <div className="mb-4 flex items-start gap-3">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
          {index + 1}
        </span>
        <div className="text-sm font-medium leading-relaxed">{q.question}</div>
      </div>

      <div className="space-y-2">
        {q.options?.map((option: string, i: number) => {
          const isSelected = selectedOption === option;
          const isCorrect = option === q.answer;
          let cls = "w-full rounded-xl border px-4 py-2.5 text-left text-sm transition-all duration-200 ";

          if (!answered) {
            cls += "border-border bg-muted/30 hover:border-primary/40 hover:bg-primary/5 cursor-pointer";
          } else if (isCorrect) {
            cls += "border-emerald-400 bg-emerald-50 text-emerald-800 font-medium";
          } else if (isSelected && !isCorrect) {
            cls += "border-red-400 bg-red-50 text-red-700";
          } else {
            cls += "border-border bg-muted/20 text-muted-foreground";
          }

          return (
            <motion.button
              key={i}
              whileHover={!answered ? { scale: 1.01, x: 2 } : {}}
              whileTap={!answered ? { scale: 0.99 } : {}}
              className={cls}
              onClick={() => !answered && onAnswer(index, option)}
              disabled={answered}
            >
              <span className="flex items-center gap-2">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-xs font-bold">
                  {String.fromCharCode(65 + i)}
                </span>
                {option}
                {answered && isCorrect && (
                  <CheckCircle2 className="ml-auto h-4 w-4 text-emerald-500" />
                )}
              </span>
            </motion.button>
          );
        })}
      </div>
    </motion.div>
  );
}

// =========================
// MARKDOWN RENDERER
// =========================

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="prose prose-sm max-w-none prose-headings:font-semibold prose-headings:text-foreground prose-p:text-foreground/80 prose-p:leading-7 prose-li:text-foreground/80 prose-strong:text-foreground prose-h1:text-xl prose-h2:text-lg prose-h3:text-base prose-ul:space-y-1 prose-ol:space-y-1">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
}

// =========================
// COMPONENT
// =========================

function VideoPage() {
  const [url, setUrl] = useState("");
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");
  const [videoData, setVideoData] = useState<any>(null);
  const [question, setQuestion] = useState("");
  const [mentorResponse, setMentorResponse] = useState("");
  const [asking, setAsking] = useState(false);
  const [currentStage, setCurrentStage] = useState(-1);
  const [summary, setSummary] = useState("");
  const [flashcards, setFlashcards] = useState<any[]>([]);
  const [quiz, setQuiz] = useState<any[]>([]);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingFlashcards, setLoadingFlashcards] = useState(false);
  const [loadingQuiz, setLoadingQuiz] = useState(false);

  // Quiz state
  const [quizAnswers, setQuizAnswers] = useState<Record<number, string>>({});
  const [quizDone, setQuizDone] = useState(false);

  // Chat history
  const [chatHistory, setChatHistory] = useState<
    { role: "user" | "ai"; text: string }[]
  >([]);

  // Flashcard navigation (mobile)
  const [cardIndex, setCardIndex] = useState(0);

  // Change 1: Session id state
  const [sessionId, setSessionId] = useState<number | null>(null);

  // Change 2: Watch time state (tracks real elapsed seconds)
  const [watchTime, setWatchTime] = useState(0);

  // Chat scroll ref
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, asking]);

  // Change 3: Track watch time locally — increments by 30 every 30 seconds
  useEffect(() => {
    if (!sessionId) return;
    const timer = setInterval(() => {
      setWatchTime((prev) => prev + 30);
    }, 30000);
    return () => clearInterval(timer);
  }, [sessionId]);

  // Change 4: Send real watch time to backend whenever watchTime updates
  useEffect(() => {
    if (!sessionId) return;
    fetch("http://127.0.0.1:8000/video/watch-time", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        watch_time: watchTime,
      }),
    });
  }, [watchTime]);

  // =========================
  // MARK COMPLETED
  // =========================

  async function markCompleted() {
    if (!sessionId) return;
    await fetch("http://127.0.0.1:8000/video/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        watch_time: 0,
      }),
    });
  }

  // =========================
  // PROCESS VIDEO
  // =========================

  async function processVideo() {
    if (!url.trim()) return;
    try {
      setProcessing(true);
      setError("");
      setVideoData(null);
      setSummary("");
      setFlashcards([]);
      setQuiz([]);
      setCurrentStage(0);
      setQuizAnswers({});
      setQuizDone(false);
      setChatHistory([]);
      setCardIndex(0);
      setSessionId(null);
      // Change 5: Reset watch time when starting a new session
      setWatchTime(0);

      const stageInterval = setInterval(() => {
        setCurrentStage((prev) => {
          if (prev >= 5) { clearInterval(stageInterval); return prev; }
          return prev + 1;
        });
      }, 1200);

      // Get logged-in user and pass user_id to backend
      const {
        data: { user },
      } = await supabase.auth.getUser();

      const response = await fetch("http://127.0.0.1:8000/video/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          youtube_url: url,
          user_id: user?.id,
        }),
      });

      const data = await response.json();

      // Surface backend errors (e.g. transcript not found, YouTube extraction failed)
      if (data.error) {
        throw new Error(data.error);
      }

      // Error check — ensure session_id was returned
      if (!data.session_id) {
        throw new Error("Session ID not returned from backend");
      }

      setVideoData(data);
      setCurrentStage(5);

      // Save session id from response
      setSessionId(data.session_id);

      // Auto-generate all content
      generateSummary();
      generateFlashcards();
      generateQuiz();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }

  // =========================
  // ASK VIDEO MENTOR
  // =========================

  async function askMentor() {
    if (!question.trim()) return;
    const userMsg = question;
    setQuestion("");
    setChatHistory((prev) => [...prev, { role: "user", text: userMsg }]);
    try {
      setAsking(true);
      const response = await fetch("http://127.0.0.1:8000/video/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg }),
      });
      const data = await response.json();
      setChatHistory((prev) => [
        ...prev,
        { role: "ai", text: data.mentor_response },
      ]);
      setMentorResponse(data.mentor_response);
    } catch (err: any) {
      setChatHistory((prev) => [
        ...prev,
        { role: "ai", text: "Error loading mentor response." },
      ]);
    } finally {
      setAsking(false);
    }
  }

  // =========================
  // GENERATE SUMMARY
  // =========================

  async function generateSummary() {
    try {
      setLoadingSummary(true);
      const response = await fetch("http://127.0.0.1:8000/video/summary");
      if (!response.ok) throw new Error("Failed");
      const data = await response.json();
      setSummary(data.summary || "");
    } catch (err: any) {
      setSummary("Error loading summary");
    } finally {
      setLoadingSummary(false);
    }
  }

  // =========================
  // GENERATE FLASHCARDS
  // =========================

  async function generateFlashcards() {
    try {
      setLoadingFlashcards(true);
      const response = await fetch("http://127.0.0.1:8000/video/flashcards");
      if (!response.ok) throw new Error("Failed");
      const data = await response.json();
      setFlashcards(data.flashcards || []);
      setCardIndex(0);
    } catch (err: any) {
      setFlashcards([]);
    } finally {
      setLoadingFlashcards(false);
    }
  }

  // =========================
  // GENERATE QUIZ
  // =========================

  async function generateQuiz() {
    try {
      setLoadingQuiz(true);
      const response = await fetch("http://127.0.0.1:8000/video/quiz");
      if (!response.ok) throw new Error("Failed");
      const data = await response.json();
      setQuiz(data.quiz || []);
      setQuizAnswers({});
      setQuizDone(false);
    } catch (err: any) {
      setQuiz([]);
    } finally {
      setLoadingQuiz(false);
    }
  }

  // =========================
  // QUIZ ANSWER HANDLER
  // =========================

  function handleQuizAnswer(qi: number, option: string) {
    const updated = { ...quizAnswers, [qi]: option };
    setQuizAnswers(updated);
    if (Object.keys(updated).length === quiz.length) {
      setQuizDone(true);
      markCompleted();
    }
  }

  const quizScore = quiz.filter((q, i) => quizAnswers[i] === q.answer).length;
  const quizPercent = quiz.length > 0 ? Math.round((quizScore / quiz.length) * 100) : 0;
  const answeredCount = Object.keys(quizAnswers).length;
  const progressPercent = quiz.length > 0 ? Math.round((answeredCount / quiz.length) * 100) : 0;

  // =========================
  // UI
  // =========================

  return (
    <PageShell
      title="Video Intelligence"
      description="Process YouTube lectures into transcripts, embeddings, quizzes, flashcards, and AI mentor support."
    >
      {/* ========================= */}
      {/* INPUT */}
      {/* ========================= */}

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card rounded-3xl p-6"
      >
        <div className="flex flex-col gap-3 md:flex-row">
          <div className="flex flex-1 items-center gap-3 rounded-2xl border border-border bg-white/80 px-4 py-3 shadow-sm transition-all focus-within:border-primary/40 focus-within:shadow-md focus-within:shadow-primary/5">
            <Youtube className="h-5 w-5 shrink-0 text-red-500" />
            <input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && processVideo()}
              placeholder="Paste YouTube URL..."
              className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            />
          </div>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={processVideo}
            disabled={processing || !url.trim()}
            className="flex items-center justify-center gap-2 rounded-2xl bg-primary px-7 py-3 font-semibold text-primary-foreground shadow-md shadow-primary/20 transition disabled:opacity-50"
          >
            {processing ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            Process
          </motion.button>
        </div>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 overflow-hidden rounded-2xl bg-red-100 p-4 text-sm text-red-600"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* ========================= */}
      {/* VIDEO EMBED */}
      {/* ========================= */}

      <AnimatePresence>
        {url && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="glass-card rounded-3xl p-6"
          >
            <div className="mb-4 text-lg font-semibold">Lecture Video</div>
            <div className="overflow-hidden rounded-2xl shadow-lg ring-1 ring-border/30">
              <iframe
                className="h-[420px] w-full"
                src={`https://www.youtube.com/embed/${getVideoId(url)}`}
                allowFullScreen
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ========================= */}
      {/* PROCESSING PIPELINE */}
      {/* ========================= */}

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card rounded-3xl p-6"
      >
        <div className="mb-5 flex items-center justify-between">
          <div className="text-lg font-semibold">Processing Pipeline</div>
          {processing && (
            <motion.div
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ repeat: Infinity, duration: 1.4 }}
              className="rounded-full bg-primary/10 px-4 py-1 text-xs font-medium text-primary"
            >
              ● Processing...
            </motion.div>
          )}
        </div>

        <div className="relative flex flex-col gap-3 md:flex-row md:items-center">
          {STAGES.map((stage, i) => {
            const Icon = stage.icon;
            const active = i === currentStage;
            const done = i < currentStage;
            const isLast = i === STAGES.length - 1;

            return (
              <div key={stage.key} className="relative flex flex-1 flex-row items-center gap-0 md:flex-col">
                {/* Connector line (between stages) */}
                {!isLast && (
                  <div className="absolute left-[calc(50%+28px)] top-[22px] hidden h-0.5 md:block"
                    style={{ right: "calc(-50% + 28px)", zIndex: 0 }}>
                    <div className="relative h-full w-full overflow-hidden rounded-full bg-border/40">
                      <motion.div
                        className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-primary/60 to-primary/30"
                        initial={{ width: "0%" }}
                        animate={{ width: done ? "100%" : active ? "50%" : "0%" }}
                        transition={{ duration: 0.8, ease: "easeInOut" }}
                      />
                    </div>
                  </div>
                )}

                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.06 }}
                  className={`relative z-10 flex flex-1 flex-col items-center rounded-2xl border p-4 text-center transition-all duration-500 ${
                    active
                      ? "border-primary/60 bg-primary/10 shadow-lg shadow-primary/15"
                      : done
                      ? "border-emerald-300 bg-emerald-50/80"
                      : "border-border bg-white/50"
                  }`}
                >
                  {/* Glow effect on active */}
                  {active && (
                    <motion.div
                      className="absolute inset-0 rounded-2xl bg-primary/8"
                      animate={{ opacity: [0, 0.8, 0] }}
                      transition={{ repeat: Infinity, duration: 1.4 }}
                    />
                  )}

                  {/* Success pulse on done */}
                  {done && (
                    <motion.div
                      className="absolute inset-0 rounded-2xl bg-emerald-400/10"
                      initial={{ scale: 1.2, opacity: 0.6 }}
                      animate={{ scale: 1, opacity: 0 }}
                      transition={{ duration: 0.5 }}
                    />
                  )}

                  <div
                    className={`relative mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-xl transition-all duration-500 ${
                      active
                        ? "bg-primary/20 text-primary shadow-md shadow-primary/20"
                        : done
                        ? "bg-emerald-100 text-emerald-600"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {active ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : done ? (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ type: "spring", stiffness: 200 }}
                      >
                        <CheckCircle2 className="h-5 w-5" />
                      </motion.div>
                    ) : (
                      <Icon className="h-5 w-5" />
                    )}
                  </div>

                  <div className="text-xs font-semibold">{stage.label}</div>
                </motion.div>
              </div>
            );
          })}
        </div>

        {!videoData && (
          <div className="mt-5 text-center text-sm text-muted-foreground">
            Process a lecture to activate the AI pipeline.
          </div>
        )}
      </motion.div>

      {/* ========================= */}
      {/* VIDEO DATA + MENTOR */}
      {/* ========================= */}

      <AnimatePresence>
        {videoData && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid gap-5 lg:grid-cols-2"
          >
            {/* VIDEO ANALYSIS */}
            <div className="glass-card rounded-3xl p-6">
              <div className="mb-5 flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary/10">
                  <Zap className="h-4 w-4 text-primary" />
                </div>
                <div className="text-lg font-semibold">Video Analysis</div>
              </div>

              {/* Video title banner */}
              <div className="mb-4 rounded-2xl border border-border/60 bg-gradient-to-r from-primary/5 to-primary/10 p-4">
                <div className="mb-1 text-xs font-medium text-muted-foreground">Video Title</div>
                <div className="text-sm font-semibold leading-snug text-foreground">
                  {videoData.video_title}
                </div>
              </div>

              {/* Stats grid */}
              <div className="mb-5 grid grid-cols-2 gap-3">
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.1 }}
                  className="rounded-2xl border border-border/60 bg-white/70 p-4 shadow-sm"
                >
                  <div className="mb-2 text-xs text-muted-foreground">Chunks Created</div>
                  <div className="flex items-end gap-2">
                    <span className="text-3xl font-black text-primary">{videoData.chunks_created}</span>
                    <Hash className="mb-1 h-4 w-4 text-primary/50" />
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.15 }}
                  className="rounded-2xl border border-emerald-200/60 bg-emerald-50/70 p-4 shadow-sm"
                >
                  <div className="mb-2 text-xs text-muted-foreground">Embeddings</div>
                  <div className="flex items-center gap-2">
                    <motion.div
                      animate={{ rotate: [0, 10, -10, 0] }}
                      transition={{ duration: 0.5, delay: 0.5 }}
                    >
                      <Sparkles className="h-5 w-5 text-emerald-500" />
                    </motion.div>
                    <span className="text-sm font-bold text-emerald-700">Success</span>
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 }}
                  className="rounded-2xl border border-emerald-200/60 bg-emerald-50/70 p-4 shadow-sm"
                >
                  <div className="mb-2 text-xs text-muted-foreground">Semantic Search</div>
                  <div className="flex items-center gap-1.5">
                    <div className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
                    <span className="text-sm font-bold text-emerald-700">Active</span>
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.25 }}
                  className="rounded-2xl border border-primary/20 bg-primary/5 p-4 shadow-sm"
                >
                  <div className="mb-2 text-xs text-muted-foreground">AI Mentor</div>
                  <div className="flex items-center gap-1.5">
                    <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
                    <span className="text-sm font-bold text-primary">Ready</span>
                  </div>
                </motion.div>
              </div>

              {/* GENERATE BUTTONS */}
              <div className="flex flex-col gap-3">
                <motion.button
                  whileHover={{ scale: 1.02, y: -1 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={generateSummary}
                  disabled={loadingSummary}
                  className="flex items-center justify-center gap-2 rounded-2xl bg-primary px-5 py-3.5 text-sm font-semibold text-primary-foreground shadow-md shadow-primary/20 transition disabled:opacity-60"
                >
                  {loadingSummary ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <FileText className="h-4 w-4" />
                  )}
                  Generate Summary
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.02, y: -1 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={generateFlashcards}
                  disabled={loadingFlashcards}
                  className="flex items-center justify-center gap-2 rounded-2xl border border-primary/30 bg-primary/5 px-5 py-3.5 text-sm font-semibold text-primary transition hover:bg-primary/10 disabled:opacity-60"
                >
                  {loadingFlashcards ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <BookOpen className="h-4 w-4" />
                  )}
                  Generate Flashcards
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.02, y: -1 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={generateQuiz}
                  disabled={loadingQuiz}
                  className="flex items-center justify-center gap-2 rounded-2xl border border-primary/30 bg-primary/5 px-5 py-3.5 text-sm font-semibold text-primary transition hover:bg-primary/10 disabled:opacity-60"
                >
                  {loadingQuiz ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Brain className="h-4 w-4" />
                  )}
                  Generate Quiz
                </motion.button>
              </div>
            </div>

            {/* AI MENTOR CHAT */}
            <div className="glass-card flex flex-col rounded-3xl p-0 overflow-hidden">
              {/* Header */}
              <div className="flex items-center gap-3 border-b border-border/50 bg-white/40 px-6 py-4 backdrop-blur-sm">
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10">
                  <MessageCircle className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <div className="text-sm font-semibold">Ask Video Mentor</div>
                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    Online · Powered by AI
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div
                className="flex-1 overflow-y-auto px-4 py-4 space-y-4"
                style={{ minHeight: 280, maxHeight: 380 }}
              >
                {chatHistory.length === 0 && (
                  <div className="flex h-full flex-col items-center justify-center gap-3 py-8">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/8">
                      <Bot className="h-6 w-6 text-primary/50" />
                    </div>
                    <div className="text-center">
                      <div className="text-sm font-medium text-foreground/60">Your AI Mentor is ready</div>
                      <div className="text-xs text-muted-foreground">Ask anything about the lecture</div>
                    </div>
                  </div>
                )}

                {chatHistory.map((msg, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 8, scale: 0.97 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.25 }}
                    className={`flex gap-2.5 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {msg.role === "ai" && (
                      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary ring-2 ring-primary/10">
                        <Bot className="h-3.5 w-3.5" />
                      </div>
                    )}

                    <div
                      className={`max-w-[78%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm ${
                        msg.role === "user"
                          ? "rounded-tr-sm bg-primary text-primary-foreground shadow-primary/20"
                          : "rounded-tl-sm border border-border/40 bg-white/80 text-foreground backdrop-blur-sm"
                      }`}
                    >
                      {msg.text}
                    </div>

                    {msg.role === "user" && (
                      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground ring-2 ring-primary/20">
                        <User className="h-3.5 w-3.5" />
                      </div>
                    )}
                  </motion.div>
                ))}

                {/* Typing indicator */}
                {asking && (
                  <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-2.5"
                  >
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary ring-2 ring-primary/10">
                      <Bot className="h-3.5 w-3.5" />
                    </div>
                    <div className="flex gap-1 rounded-2xl rounded-tl-sm border border-border/40 bg-white/80 px-4 py-3 shadow-sm backdrop-blur-sm">
                      {[0, 1, 2].map((dot) => (
                        <motion.div
                          key={dot}
                          className="h-2 w-2 rounded-full bg-primary/50"
                          animate={{ y: [0, -5, 0], opacity: [0.5, 1, 0.5] }}
                          transition={{
                            repeat: Infinity,
                            duration: 0.8,
                            delay: dot * 0.18,
                          }}
                        />
                      ))}
                    </div>
                  </motion.div>
                )}

                <div ref={chatEndRef} />
              </div>

              {/* Input bar */}
              <div className="border-t border-border/40 bg-white/40 p-4 backdrop-blur-sm">
                <div className="flex gap-2 items-center rounded-2xl border border-border/60 bg-white/80 px-3 py-2 shadow-sm focus-within:border-primary/40 focus-within:shadow-md focus-within:shadow-primary/5 transition-all">
                  <input
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && askMentor()}
                    placeholder="Ask about the lecture..."
                    className="flex-1 bg-transparent py-1.5 text-sm outline-none placeholder:text-muted-foreground/60"
                  />
                  <motion.button
                    whileHover={{ scale: 1.08 }}
                    whileTap={{ scale: 0.92 }}
                    onClick={askMentor}
                    disabled={asking || !question.trim()}
                    className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm shadow-primary/20 disabled:opacity-40 transition"
                  >
                    <Send className="h-3.5 w-3.5" />
                  </motion.button>
                </div>
                <div className="mt-1.5 text-center text-[10px] text-muted-foreground/50">Press Enter to send</div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ========================= */}
      {/* SUMMARY */}
      {/* ========================= */}

      <AnimatePresence>
        {summary && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="glass-card rounded-3xl p-6"
          >
            <div className="mb-5 flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary/10">
                <FileText className="h-4 w-4 text-primary" />
              </div>
              <div>
                <div className="text-lg font-semibold">AI Lecture Summary</div>
                <div className="text-xs text-muted-foreground">AI-generated overview of the lecture content</div>
              </div>
            </div>

            <div
              className="overflow-y-auto rounded-2xl border border-border/40 bg-white/60 px-6 py-5 backdrop-blur-sm"
              style={{ maxHeight: 420 }}
            >
              <MarkdownContent content={summary} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ========================= */}
      {/* FLASHCARDS */}
      {/* ========================= */}

      <AnimatePresence>
        {flashcards.length === 0 && videoData && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="glass-card rounded-3xl p-6 text-center text-sm text-muted-foreground"
          >
            No flashcards generated yet.
          </motion.div>
        )}
        {flashcards.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="glass-card rounded-3xl p-6"
          >
            <div className="mb-5 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary/10">
                  <BookOpen className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <div className="text-lg font-semibold">AI Flashcards</div>
                  <div className="text-xs text-muted-foreground">Click any card to flip</div>
                </div>
              </div>
              <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                {flashcards.length} cards
              </span>
            </div>

            {/* Desktop grid */}
            <div className="hidden gap-4 md:grid md:grid-cols-2 lg:grid-cols-3">
              {flashcards.map((card, i) => (
                <FlipCard key={i} card={card} index={i} />
              ))}
            </div>

            {/* Mobile carousel */}
            <div className="md:hidden">
              <AnimatePresence mode="wait">
                <motion.div
                  key={cardIndex}
                  initial={{ opacity: 0, x: 30 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -30 }}
                  transition={{ duration: 0.22 }}
                >
                  <FlipCard card={flashcards[cardIndex]} index={cardIndex} />
                </motion.div>
              </AnimatePresence>

              <div className="mt-4 flex items-center justify-between">
                <motion.button
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setCardIndex((i) => Math.max(0, i - 1))}
                  disabled={cardIndex === 0}
                  className="flex items-center gap-1.5 rounded-xl border border-border bg-white/70 px-4 py-2 text-sm font-medium disabled:opacity-40 hover:border-primary/30 transition"
                >
                  <ChevronLeft className="h-4 w-4" /> Previous
                </motion.button>

                <span className="text-sm font-medium text-muted-foreground">
                  {cardIndex + 1} / {flashcards.length}
                </span>

                <motion.button
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setCardIndex((i) => Math.min(flashcards.length - 1, i + 1))}
                  disabled={cardIndex === flashcards.length - 1}
                  className="flex items-center gap-1.5 rounded-xl border border-border bg-white/70 px-4 py-2 text-sm font-medium disabled:opacity-40 hover:border-primary/30 transition"
                >
                  Next <ChevronRight className="h-4 w-4" />
                </motion.button>
              </div>

              {/* Dot indicators */}
              <div className="mt-3 flex justify-center gap-1.5">
                {flashcards.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setCardIndex(i)}
                    className={`h-1.5 rounded-full transition-all ${
                      i === cardIndex ? "w-5 bg-primary" : "w-1.5 bg-border"
                    }`}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ========================= */}
      {/* QUIZ */}
      {/* ========================= */}

      <AnimatePresence>
        {quiz.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="glass-card rounded-3xl p-6"
          >
            {/* Quiz header */}
            <div className="mb-5 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary/10">
                  <Brain className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <div className="text-lg font-semibold">AI Quiz</div>
                  <div className="text-xs text-muted-foreground">{quiz.length} questions</div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                  {answeredCount}/{quiz.length} answered
                </span>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => {
                    setQuizAnswers({});
                    setQuizDone(false);
                  }}
                  className="flex items-center gap-1 rounded-full border border-border bg-white/60 px-3 py-1 text-xs font-medium hover:border-primary/30 transition"
                >
                  <RotateCcw className="h-3 w-3" />
                  Reset
                </motion.button>
              </div>
            </div>

            {/* Progress bar */}
            <div className="mb-5">
              <div className="mb-1.5 flex items-center justify-between text-xs text-muted-foreground">
                <span>Progress</span>
                <span>{progressPercent}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-muted/50">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-primary to-primary/70"
                  initial={{ width: "0%" }}
                  animate={{ width: `${progressPercent}%` }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                />
              </div>
            </div>

            {/* SCORE BANNER */}
            <AnimatePresence>
              {quizDone && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95, y: -10 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="mb-6 overflow-hidden rounded-2xl border border-amber-200/60 bg-gradient-to-r from-amber-50 via-yellow-50 to-orange-50 shadow-md shadow-amber-100"
                >
                  <div className="flex items-center gap-5 p-5">
                    <motion.div
                      animate={{ rotate: [0, -10, 10, -5, 5, 0] }}
                      transition={{ duration: 0.6, delay: 0.2 }}
                    >
                      <Trophy className="h-10 w-10 shrink-0 text-amber-500" />
                    </motion.div>
                    <div className="flex-1">
                      <div className="text-xl font-bold text-amber-900">
                        {quizScore}/{quiz.length} Correct
                      </div>
                      <div className="mt-0.5 text-sm text-amber-700/80">
                        {quizPercent >= 90
                          ? "Excellent understanding 🎉"
                          : quizPercent >= 70
                          ? "Good work 👍"
                          : quizPercent >= 50
                          ? "Keep practising 📚"
                          : "Review the lecture and try again 🔄"}
                      </div>
                      {/* Score bar */}
                      <div className="mt-2.5 h-1.5 w-full overflow-hidden rounded-full bg-amber-200/50">
                        <motion.div
                          className={`h-full rounded-full ${
                            quizPercent === 100
                              ? "bg-emerald-500"
                              : quizPercent >= 70
                              ? "bg-amber-500"
                              : "bg-red-400"
                          }`}
                          initial={{ width: "0%" }}
                          animate={{ width: `${quizPercent}%` }}
                          transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
                        />
                      </div>
                    </div>
                    <div className="flex flex-col items-center">
                      <div className="text-4xl font-black text-amber-500">{quizPercent}%</div>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => {
                          setQuizAnswers({});
                          setQuizDone(false);
                        }}
                        className="mt-2 flex items-center gap-1 rounded-xl bg-amber-100 px-3 py-1.5 text-xs font-semibold text-amber-700 hover:bg-amber-200 transition"
                      >
                        <RotateCcw className="h-3 w-3" /> Retake
                      </motion.button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="space-y-4">
              {quiz.map((q, i) => (
                <QuizQuestion
                  key={i}
                  q={q}
                  index={i}
                  onAnswer={handleQuizAnswer}
                  answered={quizAnswers[i] !== undefined}
                  selectedOption={quizAnswers[i] ?? null}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </PageShell>
  );
}